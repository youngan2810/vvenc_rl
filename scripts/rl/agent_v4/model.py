"""Spatial categorical PPO with independent actor/critic and task auxiliary head."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from agent_v4 import MODEL_VERSION
from agent_v4.state import geometry, GLOBAL_DIM, LOCAL_DIM, HISTORY, HISTORY_DIM
from agent_v4.learning import episode_scores, counterfactual_advantages, merge_actor_gradients
from ppo_agent import quantized_action_probabilities, PPOActorCritic


@dataclass(frozen=True)
class ModelConfig:
    rich_state: bool = True
    content: bool = True
    attention: bool = True


class Tower(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.local = nn.Sequential(nn.Conv2d(LOCAL_DIM, 32, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(GLOBAL_DIM+HISTORY*HISTORY_DIM, 64),
                                     nn.LayerNorm(64), nn.SiLU())
        if config.content:
            self.image = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),
                                       nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(32+64+3+(32 if config.content else 0), 64),
                                  nn.LayerNorm(64), nn.SiLU())
        self.register_buffer("geometry", geometry(), persistent=True)
        if config.attention:
            self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
            self.norm = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
            self.norm2 = nn.LayerNorm(64)

    def forward(self, s):
        local = self.local(s["local"]).flatten(2).transpose(1, 2)
        context = self.context(torch.cat((s["global"], s["history"].flatten(1)), 1))
        parts = [local, context[:, None].expand(-1, 12, -1),
                 self.geometry[None].expand(local.shape[0], -1, -1)]
        if self.config.content:
            # Y'=16..235 TV range; fixed scaling, not dataset normalization.
            image = self.image(((s["image"].float()-16)/219).clamp(0, 1))
            # Image was reduced 2x; CNN strides 4x => original CTU128 ->16.
            # Last column is12 and last row8 feature pixels, not equal bins.
            cells = [image[:, :, y:y+16, x:x+16].mean((2, 3))
                     for y in (0, 16, 32) for x in (0, 16, 32, 48)]
            parts.append(torch.stack(cells, 1))
        tokens = self.fuse(torch.cat(parts, -1))
        if self.config.attention:
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm(tokens+attended)
            tokens = self.norm2(tokens+self.ff(tokens))
        return tokens


class Agent(nn.Module):
    model_version = MODEL_VERSION

    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        if not config.rich_state and not config.content and not config.attention:
            # Action-representation control: exact v3 backbone/critic and
            # exact44 observations. Only replace Gaussian/quantization head.
            self.control = PPOActorCritic()
            del self.control.actor_log_std
            self.control.actor_cells[-1] = nn.Conv2d(32, 5, 1)
            self.occupancy_head = nn.Linear(32, 1)
            self.outcome_head = nn.Sequential(nn.Linear(32+60,64),nn.SiLU(),nn.Linear(64,3))
            self.counterfactual_head = nn.Sequential(nn.Linear(128+60,128),nn.SiLU(),nn.Linear(128,5))
            nn.init.zeros_(self.counterfactual_head[-1].weight)
            nn.init.zeros_(self.counterfactual_head[-1].bias)
            prior = quantized_action_probabilities(torch.zeros(1), torch.ones(1)*math.exp(-.7))[0]
            nn.init.orthogonal_(self.control.actor_cells[-1].weight, gain=.01)
            with torch.no_grad():
                self.control.actor_cells[-1].bias.copy_(prior.log())
            return
        self.actor = Tower(config)
        self.critic = Tower(config)  # no shared parameters / representation
        self.action_head = nn.Linear(64, 5)
        self.occupancy_head = nn.Linear(64, 1)
        self.outcome_head = nn.Sequential(nn.Linear(64+60,64),nn.SiLU(),nn.Linear(64,3))
        self.counterfactual_head = nn.Sequential(nn.Linear(128+60,128),nn.SiLU(),nn.Linear(128,5))
        nn.init.zeros_(self.counterfactual_head[-1].weight)
        nn.init.zeros_(self.counterfactual_head[-1].bias)
        self.value_head = nn.Sequential(nn.Linear(12*64, 128), nn.SiLU(), nn.Linear(128, 1))
        # Match v3's initial *quantized* distribution, rather than uniform.
        prior = quantized_action_probabilities(torch.zeros(1), torch.ones(1)*math.exp(-.7))[0]
        nn.init.orthogonal_(self.action_head.weight, gain=.01)
        with torch.no_grad():
            self.action_head.bias.copy_(prior.log())
        nn.init.zeros_(self.value_head[-1].bias)

    def actor_parameters(self):
        if hasattr(self, "control"):
            return self.control.actor_parameters()+list(self.occupancy_head.parameters())+list(self.outcome_head.parameters())
        return (list(self.actor.parameters())+list(self.action_head.parameters())+
                list(self.occupancy_head.parameters())+list(self.outcome_head.parameters()))

    def critic_parameters(self):
        if hasattr(self, "control"):
            return self.control.critic_parameters()+list(self.counterfactual_head.parameters())
        return list(self.critic.parameters())+list(self.value_head.parameters())+list(self.counterfactual_head.parameters())

    def counterfactual_values(self, s, actions):
        """Q_i(h,a_-i)[5]. Own action is masked BEFORE any action mixing."""
        if actions.ndim != 2 or actions.shape[1] != 12 or not bool(((actions >= -2)&(actions <= 2)).all()):
            raise ValueError("Invalid counterfactual actions")
        if hasattr(self,"control"):
            features = self.control.critic_encoder(torch.cat((s['local'][:,:3].flatten(1),s['global'][:,:8]),1))
            features = features[:,None].expand(-1,12,-1)
        else:
            tokens = self.critic(s)
            features = torch.cat((tokens,tokens.mean(1)[:,None].expand(-1,12,-1)),-1)
        onehot = F.one_hot((actions+2).long(),5).to(features.dtype)
        mask = 1-torch.eye(12,device=actions.device,dtype=features.dtype)
        others = (onehot[:,None]*mask[None,:,:,None]).flatten(2)
        return self.counterfactual_head(torch.cat((features,others),-1))

    def task_outcomes(self, tokens, actions):
        """Action-conditioned log1p local TP/FP/FN prediction, training only."""
        joint = F.one_hot((actions+2).long(),5).to(tokens.dtype).flatten(1)
        return self.outcome_head(torch.cat((tokens,joint[:,None].expand(-1,12,-1)),-1))

    def policy_features(self, s):
        if hasattr(self, "control"):
            m = self.control
            spatial, global_ = s["local"][:, :3], s["global"][:, :8]
            features = m.spatial_norm(m.spatial_encoder(spatial))
            context = m.fusion(torch.cat((features.mean((2, 3)), m.global_encoder(global_)), 1))
            hidden = m.actor_cells[:2](torch.cat((features,
                context[:, :, None, None].expand(-1,-1,3,4),
                m.grid_xy.expand(spatial.shape[0],-1,-1,-1)), 1))
            logits = m.actor_cells[-1](hidden).flatten(2).transpose(1, 2)
            return Categorical(logits=logits), hidden.flatten(2).transpose(1,2)
        tokens = self.actor(s)
        return Categorical(logits=self.action_head(tokens)), tokens

    def forward(self, s):
        distribution, tokens = self.policy_features(s)
        if hasattr(self, "control"):
            m = self.control
            value = m.critic(m.critic_encoder(torch.cat((s["local"][:,:3].flatten(1),s["global"][:,:8]),1))).squeeze(-1)
        else:
            value = self.value_head(self.critic(s).flatten(1)).squeeze(-1)
        return distribution, value, self.occupancy_head(tokens).squeeze(-1)

    def sample(self, s, deterministic=False, actor_only=False):
        if actor_only:
            d, _ = self.policy_features(s)
            value = d.probs.new_zeros(d.probs.shape[0])
        else:
            d, value, _ = self(s)
        indices = d.probs.argmax(-1) if deterministic else d.sample()
        return indices-2, d.log_prob(indices).sum(-1), value, d.probs


@dataclass(frozen=True)
class PPOConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip: float = .2
    epochs: int = 4
    minibatch: int = 32
    target_kl: float = .03
    entropy: float = .01
    auxiliary: float = .1
    max_grad_norm: float = .5
    credit_assignment: str = "counterfactual-final"
    auxiliary_mode: str = "task-outcome"
    auxiliary_grad_ratio: float = .5
    paired_baseline: bool = False  # production clip-delta trainer sets True

    def __post_init__(self):
        if (self.epochs < 1 or self.minibatch < 1 or
                any(not math.isfinite(x) or x <= 0 for x in
                    (self.actor_lr, self.critic_lr, self.clip, self.target_kl, self.max_grad_norm)) or
                not 0 < self.clip < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or
                self.credit_assignment not in ('counterfactual-final','shared-final','joint-gae') or
                self.auxiliary_mode not in ('task-outcome','presence') or
                (self.credit_assignment != 'joint-gae' and (self.gamma != 1 or self.gae_lambda != 1)) or
                (self.paired_baseline and self.credit_assignment == 'joint-gae') or
                any(not math.isfinite(x) or x < 0 for x in (self.entropy, self.auxiliary,self.auxiliary_grad_ratio))):
            raise ValueError("Invalid PPO hyperparameters")


def advantages(rewards, values, dones, gamma=1., lam=1.):
    if len(rewards) == 0 or not bool(dones[-1]):
        raise ValueError("Only complete episodes are accepted; no implicit bootstrap")
    result = torch.zeros_like(rewards)
    future = rewards.new_zeros(())
    for i in range(len(rewards)-1, -1, -1):
        mask = 1.-dones[i].float()
        next_value = values[i+1] if i+1 < len(values) else values.new_zeros(())
        delta = rewards[i]+gamma*mask*next_value-values[i]
        future = delta+gamma*lam*mask*future
        result[i] = future
    return result, result+values


class Trainer:
    def __init__(self, model: Agent, config=PPOConfig()):
        self.model, self.config = model, config
        self.actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=config.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=config.critic_lr, eps=1e-5)

    @torch.no_grad()
    def parity(self, state, actions, old_logp, old_values):
        n = len(actions)
        if (n < 1 or actions.shape != (n,12) or actions.dtype not in (torch.int32,torch.int64) or
                not bool(((actions >= -2) & (actions <= 2)).all())):
            raise ValueError("Expected a nonempty integer delta-QP batch [N,12]")
        for name,t in (("old_logp",old_logp),("old_values",old_values)):
            if t.shape != (n,) or not torch.isfinite(t).all():
                raise ValueError("Invalid/non-finite " + name)
        expected = {"local":(n,LOCAL_DIM,3,4), "global":(n,GLOBAL_DIM),
                    "history":(n,HISTORY,HISTORY_DIM), "image":(n,1,160,240)}
        if set(state) != set(expected):
            raise ValueError("Missing/unexpected state channels")
        for name,shape in expected.items():
            t = state[name]
            if tuple(t.shape) != shape or (t.is_floating_point() and not torch.isfinite(t).all()):
                raise ValueError("Invalid/non-finite state: " + name)
        if state["image"].dtype != torch.uint8:
            raise ValueError("Source thumbnail must be uint8")
        errors = [0., 0.]
        for i in range(0, len(actions), self.config.minibatch):
            sl = slice(i, i+self.config.minibatch)
            d, v, _ = self.model({k:t[sl] for k,t in state.items()})
            if not torch.isfinite(v).all() or not torch.isfinite(d.logits).all():
                raise ValueError("Non-finite collecting policy/value")
            errors[0] = max(errors[0], float((d.log_prob(actions[sl]+2).sum(-1)-old_logp[sl]).abs().max()))
            errors[1] = max(errors[1], float((v-old_values[sl]).abs().max()))
        if max(errors) > 2e-4:
            raise ValueError(f"Off-policy/state parity failure: {errors}")
        return errors

    @torch.no_grad()
    def policy_log_probs(self, state):
        size = self.config.minibatch
        return torch.cat([self.model.policy_features({k:t[i:i+size] for k,t in state.items()})[0].logits
                          for i in range(0,len(state["global"]),size)])

    def update(self, state, actions, old_logp, old_values, rewards, dones, occupancy,
               task_outcomes=None, baseline_scores=None):
        c = self.config
        parity = self.parity(state, actions, old_logp, old_values)
        n = len(actions)
        if (rewards.shape != (n,) or not torch.isfinite(rewards).all() or
                dones.shape != (n,) or dones.dtype != torch.bool or
                occupancy.shape != (n,12) or not torch.isfinite(occupancy).all() or
                not bool(((occupancy >= 0) & (occupancy <= 1)).all())):
            raise ValueError("Invalid reward/boundary/teacher batch")
        if c.auxiliary and c.auxiliary_mode == 'task-outcome':
            if (task_outcomes is None or task_outcomes.shape != (n,12,3) or
                    not torch.isfinite(task_outcomes).all() or not bool((task_outcomes >= 0).all())):
                raise ValueError("Task auxiliary requires measured local TP/FP/FN [N,12,3]")
        final_credit = c.credit_assignment != 'joint-gae'
        if final_credit:
            returns = episode_scores(rewards,dones)
            if c.paired_baseline:
                if (baseline_scores is None or baseline_scores.shape != (n,) or
                        not torch.isfinite(baseline_scores).all()):
                    raise ValueError("Paired training requires frozen vanilla scores")
                first = 0
                for end in torch.where(dones)[0].tolist():
                    if not torch.equal(baseline_scores[first:end+1],baseline_scores[first].expand(end+1-first)):
                        raise ValueError("Baseline score must be constant within its clip")
                    first = end+1
                returns = returns-baseline_scores
            # Freeze BEFORE fitting this batch: no chosen-action target can
            # leak into the current actor baseline through a critic update.
            with torch.no_grad():
                probs = self.policy_log_probs(state).exp()
                if c.credit_assignment == 'counterfactual-final':
                    q = torch.cat([self.model.counterfactual_values(
                        {k:t[i:i+c.minibatch] for k,t in state.items()},actions[i:i+c.minibatch])
                        for i in range(0,n,c.minibatch)])
                    adv, cf_baseline = counterfactual_advantages(returns,probs,q)
                else:
                    cf_baseline = old_values[:,None].expand(-1,12)
                    adv = returns[:,None]-cf_baseline
        else:
            adv, returns = advantages(rewards, old_values, dones, c.gamma, c.gae_lambda)
        advantage_std = float(adv.std(unbiased=False))
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        actor_stopped, steps, actor_steps = False, 0, 0
        stats = {k:[] for k in ('policy_loss','value_loss','final_value_loss','counterfactual_value_loss',
                'entropy','auxiliary_loss','kl','clip_fraction',
                'primary_grad_norm','auxiliary_grad_norm','auxiliary_grad_cosine',
                'auxiliary_grad_scale','auxiliary_grad_conflict')}
        actor_params = self.model.actor_parameters()
        actor_before = [p.detach().clone() for p in actor_params]
        optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
        # Keep old probabilities for exact factorized categorical KL. Their
        # sum over CTUs is the KL of the conditional product distribution.
        with torch.no_grad():
            old_log_probs = self.policy_log_probs(state)
            old_probs = old_log_probs.exp()
        for _ in range(c.epochs):
            for idx in torch.randperm(len(actions)).split(c.minibatch):
                batch = {k:t[idx] for k,t in state.items()}
                d,tokens = self.model.policy_features(batch)
                # independent critic graph; actor/aux cannot train critic
                _,value,aux = self.model(batch)
                head_logp = d.log_prob(actions[idx]+2)
                logp = head_logp if final_credit else head_logp.sum(-1)
                old = old_log_probs[idx].gather(-1,(actions[idx]+2).unsqueeze(-1)).squeeze(-1) if final_credit else old_logp[idx]
                log_ratio = logp-old
                ratio = log_ratio.exp()
                surrogate = torch.minimum(ratio*adv[idx], ratio.clamp(1-c.clip, 1+c.clip)*adv[idx])
                policy_loss = -(surrogate.sum(-1).mean() if final_credit else surrogate.mean())
                # Report/regularize mean per-CTU entropy. Joint logprob and
                # joint KL remain SUMS. This fixes coefficient units at12 heads.
                entropy = d.entropy().mean()
                # Presence supervision gives small car boxes a useful signal
                # even if they occupy a tiny fraction of a large CTU. Area
                # targets are archived separately; this is NOT per-cell F1.
                if c.auxiliary_mode == 'task-outcome' and c.auxiliary:
                    aux_loss = F.smooth_l1_loss(self.model.task_outcomes(tokens,actions[idx]),
                                              torch.log1p(task_outcomes[idx]))
                else:
                    aux_loss = F.binary_cross_entropy_with_logits(aux, (occupancy[idx] > 0).float(),
                        pos_weight=aux.new_tensor(3.))
                final_value_loss = .5*F.mse_loss(value, returns[idx])
                cf_value_loss = value.new_zeros(())
                if c.credit_assignment == 'counterfactual-final':
                    chosen = self.model.counterfactual_values(batch,actions[idx]).gather(-1,(actions[idx]+2).unsqueeze(-1)).squeeze(-1)
                    cf_value_loss = .5*F.mse_loss(chosen,returns[idx,None].expand_as(chosen))
                value_loss = final_value_loss+cf_value_loss
                with torch.no_grad():
                    kl = (old_probs[idx]*(old_log_probs[idx]-d.logits)).sum((1, 2)).mean().clamp_min(0)
                if kl > c.target_kl:
                    actor_stopped = True
                if not actor_stopped:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss = policy_loss-c.entropy*entropy
                    if not torch.isfinite(actor_loss):
                        raise ValueError("Non-finite actor loss")
                    primary = torch.autograd.grad(actor_loss,actor_params,allow_unused=True,retain_graph=bool(c.auxiliary))
                    auxiliary = (torch.autograd.grad(c.auxiliary*aux_loss,actor_params,allow_unused=True)
                                 if c.auxiliary else [None]*len(actor_params))
                    merged,diagnostics = merge_actor_gradients(primary,auxiliary,actor_params,c.auxiliary_grad_ratio)
                    for p,g in zip(actor_params,merged):
                        p.grad = g
                    for key,val in diagnostics.items():
                        stats[key].append(val)
                    nn.utils.clip_grad_norm_(self.model.actor_parameters(), c.max_grad_norm, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    actor_steps += 1
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.model.critic_parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
                steps += 1
                for key, val in (("policy_loss", policy_loss), ("value_loss", value_loss),
                                 ('final_value_loss',final_value_loss),('counterfactual_value_loss',cf_value_loss),
                                 ("entropy", entropy), ("auxiliary_loss", aux_loss), ("kl", kl),
                                 ("clip_fraction", ((ratio-1).abs() > c.clip).float().mean())):
                    stats[key].append(float(val.detach()))
        with torch.no_grad():
            counts = torch.bincount((actions+2).flatten(), minlength=5).float()
            def measure():
                log_probs = self.policy_log_probs(state)
                kl = (old_probs*(old_log_probs-log_probs)).sum((1,2)).mean().clamp_min(0)
                return log_probs, float(kl)
            final_log_probs, exact_kl = measure()
            proposed_kl = exact_kl
            accepted_scale = 1.
            # PPO clipping is not a hard KL bound. Check ALL rollout states,
            # including after the last actor/aux step. Backtrack parameters
            # if necessary and reset Adam moments to the pre-update state.
            # This is an empirical rollout guard, not a population guarantee.
            if not math.isfinite(exact_kl) or exact_kl > c.target_kl:
                actor_stopped = True
                proposed = [p.detach().clone() for p in actor_params]
                for power in range(1,11):
                    accepted_scale = .5**power
                    for p,before,after in zip(actor_params,actor_before,proposed):
                        p.copy_(before+accepted_scale*(after-before))
                    final_log_probs,exact_kl = measure()
                    if math.isfinite(exact_kl) and exact_kl <= c.target_kl:
                        break
                else:
                    accepted_scale = 0.
                    for p,before in zip(actor_params,actor_before):
                        p.copy_(before)
                    final_log_probs,exact_kl = measure()
                self.actor_optimizer.load_state_dict(optimizer_before)
            final_probs = final_log_probs.exp()
        return {**{k:sum(v)/max(1,len(v)) for k,v in stats.items()},
                "credit_assignment":c.credit_assignment,"auxiliary_mode":c.auxiliary_mode,
                "paired_baseline":c.paired_baseline,
                "counterfactual_baseline_std":float(cf_baseline.std(unbiased=False)) if final_credit else None,
                'frozen_final_value_mse':float((returns-old_values).square().mean()),
                'frozen_chosen_q_mse':float((q.gather(-1,(actions+2).unsqueeze(-1)).squeeze(-1)-returns[:,None]).square().mean())
                    if c.credit_assignment == 'counterfactual-final' else None,
                'observed_target_mean':float(returns.mean()),
                "exact_final_kl": exact_kl, "actor_early_stop": actor_stopped,
                "proposed_final_kl": proposed_kl if math.isfinite(proposed_kl) else None,
                "accepted_actor_scale": accepted_scale,
                "actor_optimizer_reset": accepted_scale < 1.,
                "optimizer_steps": steps, "actor_steps": actor_steps,
                "collector_parity": parity, "mean_reward": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "return_std": float(returns.std(unbiased=False)), "advantage_std": advantage_std,
                "teacher_positive_ctu_fraction": float((occupancy > 0).float().mean()),
                "action_frequencies": (counts/counts.sum()).tolist(),
                "deterministic_nonzero_fraction": float((final_probs.argmax(-1) != 2).float().mean()),
                "expected_sampled_nonzero_fraction": float((1-final_probs[:, :, 2]).mean())}

