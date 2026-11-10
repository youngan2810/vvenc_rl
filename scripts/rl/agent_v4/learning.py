"""Explicit final-score credit and bounded auxiliary gradients for v4.2.

Only complete fresh trajectories are accepted. Counterfactual predictions are
baselines, never invented rewards/labels for actions the codec did not execute.
"""
import math
import torch


def episode_scores(rewards, dones):
    """Broadcast each COMPLETE episode's final sum, not its prefix tail return."""
    if (rewards.ndim != 1 or dones.shape != rewards.shape or dones.dtype != torch.bool or
            not len(rewards) or not bool(dones[-1]) or not torch.isfinite(rewards).all()):
        raise ValueError("Final-score targets require finite complete episodes")
    result = torch.empty_like(rewards)
    first = 0
    for end in torch.where(dones)[0].tolist():
        result[first:end+1] = rewards[first:end+1].sum()
        first = end+1
    return result


def counterfactual_advantages(final_scores, old_probs, frozen_q):
    """Observed outcome minus an own-action-independent frozen baseline.

    Q_i must be computed with action i MASKED. For conditional product policies
    E[grad log pi_i * b_i(h,a_-i)] = 0, even for an imperfect frozen critic.
    Never replace the observed outcome by Q(chosen) without acknowledging bias.
    """
    n = len(final_scores)
    if (final_scores.shape != (n,) or old_probs.shape != (n,12,5) or
            frozen_q.shape != old_probs.shape or
            not all(torch.isfinite(t).all() for t in (final_scores,old_probs,frozen_q)) or
            not bool((old_probs >= 0).all()) or
            not torch.allclose(old_probs.sum(-1),old_probs.new_ones(n,12),atol=2e-6)):
        raise ValueError("Invalid frozen counterfactual baseline")
    baseline = (old_probs.detach()*frozen_q.detach()).sum(-1)
    return final_scores[:,None]-baseline, baseline


def merge_actor_gradients(primary, auxiliary, parameters, ratio):
    """Project conflicting auxiliary gradient, then cap its GLOBAL L2 norm.

    These diagnostics concern raw parameter gradients, not Adam's resulting
    step or a guarantee of population return improvement. KL is checked later.
    """
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("Invalid auxiliary gradient budget")
    zero = parameters[0].new_zeros(())
    main2 = sum((g.square().sum() for g in primary if g is not None), zero)
    aux2 = sum((g.square().sum() for g in auxiliary if g is not None), zero)
    dot = sum(((a*b).sum() for a,b in zip(primary,auxiliary)
               if a is not None and b is not None), zero)
    if not all(torch.isfinite(x) for x in (main2,aux2,dot)):
        raise ValueError("Non-finite actor gradient")
    cosine = float(dot/(main2.sqrt()*aux2.sqrt()+1e-20))
    conflict = bool(dot < 0)
    projected = []
    for a,b,p in zip(primary,auxiliary,parameters):
        g = torch.zeros_like(p) if b is None else b
        if conflict and a is not None:
            g = g-dot/(main2+1e-20)*a
        projected.append(g)
    norm = sum((g.square().sum() for g in projected), zero).sqrt()
    scale = min(1.,float(ratio*main2.sqrt()/(norm+1e-20)))
    # An exactly absent primary learning signal cannot be replaced by aux-only
    # updates masquerading as reinforcement learning.
    if float(main2) < 1e-20:
        scale = 0.
    merged = [(torch.zeros_like(p) if a is None else a)+scale*b
              for a,b,p in zip(primary,projected,parameters)]
    return merged, {"primary_grad_norm":float(main2.sqrt()),
                    "auxiliary_grad_norm":float(aux2.sqrt()),
                    "auxiliary_grad_cosine":cosine,
                    "auxiliary_grad_scale":scale,
                    "auxiliary_grad_conflict":float(conflict)}
