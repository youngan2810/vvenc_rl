"""Regression gates for failure modes observed in the real EP467 audit."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import copy

import torch

from ppo_agent import PPOActorCritic,PPOTrainer,Transition,quantized_action_probabilities
from full_train import create_schedule,atomic_checkpoint,train_checkpoint,load_trainer
from ppo_agent_legacy import PPOActorCritic as LegacyModel,PPOTrainer as LegacyTrainer


class RestartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_quantized_probabilities_and_entropy_gradient_match_actual_actions(self):
        mean=torch.tensor([[-2.,0.,1.1]],requires_grad=True)
        std=torch.full_like(mean,.5)
        p=quantized_action_probabilities(mean,std)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones_like(mean)))
        samples=(2*torch.distributions.Normal(mean.detach(),std).sample((100000,)).tanh()).round()
        empirical=torch.stack([(samples==k).float().mean(0) for k in range(-2,3)],-1)
        self.assertTrue(torch.allclose(p,empirical,atol=.006))
        entropy=-(p*p.clamp_min(1e-30).log()).sum()
        entropy.backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        # Ascending actual entropy must pull a saturated negative mean inward.
        self.assertGreater(float(mean.grad[0,0]),0.)

    def test_quantized_probability_extreme_tails_remain_finite(self):
        mean=torch.tensor([[-100.,100.,-4.,4.]],requires_grad=True)
        std=torch.full_like(mean,.1)
        p=quantized_action_probabilities(mean,std)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones_like(mean)))
        (-(p*p.clamp_min(1e-30).log()).sum()).backward()
        self.assertTrue(torch.isfinite(mean.grad).all())

    def test_actor_and_critic_features_and_gradients_are_independent(self):
        model=PPOActorCritic();sp=torch.rand(32,3,3,4);gl=torch.rand(32,8)
        self.assertFalse({id(p) for p in model.actor_parameters()} & {id(p) for p in model.critic_parameters()})
        self.assertEqual(len(model.actor_parameters())+len(model.critic_parameters()),len(list(model.parameters())))
        before=model(sp,gl)[0].mean.detach().clone()
        model(sp,gl)[1].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.actor_parameters()))
        with torch.no_grad():
            for p in model.critic_parameters():p.add_(torch.randn_like(p)*.1)
        self.assertTrue(torch.equal(before,model(sp,gl)[0].mean.detach()))
        self.assertGreater(float(model(sp,gl)[1].detach().std()),.01)

    def test_fresh_policy_starts_near_vanilla_and_preserves_spatial_response(self):
        model=PPOActorCritic();sp=torch.rand(64,3,3,4);gl=torch.rand(64,8)
        raw,actions,_,_=model.sample(sp,gl,deterministic=True)
        self.assertTrue(torch.equal(actions,torch.zeros_like(actions)))
        self.assertLess(float(raw.detach().abs().max()),.1)
        changed=sp.clone();changed[:,0,0,0]+=.5
        self.assertGreater(float((model(changed,gl)[0].mean-model(sp,gl)[0].mean).detach().abs().max()),1e-5)

    def test_critic_learns_state_dependent_returns_while_kl_freezes_actor(self):
        # Synthetic supervised-return gate; NOT evidence of VVenC task gains.
        model=PPOActorCritic();sp=torch.rand(64,3,3,4);gl=torch.rand(64,8)
        with torch.no_grad():raw,_,lp,value=model.sample(sp,gl)
        target=.3+10*gl[:,0]
        rows=[Transition(sp[i],gl[i],raw[i],float(lp[i])+1.,float(target[i]),float(value[i]),0.,True) for i in range(64)]
        trainer=PPOTrainer(model,epochs=2,minibatch_size=32)
        actor_before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('critic')}
        initial=float((value-target).square().mean())
        for _ in range(50):
            report=trainer.update(rows)
            self.assertEqual(report['optimizer_steps'],0)
            self.assertEqual(report['critic_optimizer_steps'],4)
        with torch.no_grad():pred=model(sp,gl)[1]
        self.assertLess(float((pred-target).square().mean()),initial*.05)
        self.assertGreater(report['critic_explained_variance'],.8)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in actor_before.items()))

    def test_all_videos_appear_before_any_video_repeats(self):
        schedule=create_schedule(20261103,2)
        for epoch in range(2):
            for round_index in range(5):
                block=schedule[epoch*325+round_index*65:epoch*325+(round_index+1)*65]
                self.assertEqual({r['train_index'] for r in block},set(range(65)))

    def test_kl_freeze_preserves_existing_actor_adam_momentum(self):
        model=PPOActorCritic();trainer=PPOTrainer(model,epochs=2,minibatch_size=16)
        sp=torch.rand(32,3,3,4);gl=torch.rand(32,8)
        def collect(log_shift):
            with torch.no_grad():raw,_,lp,value=model.sample(sp,gl)
            return [Transition(sp[i],gl[i],raw[i],float(lp[i])+log_shift,.2+i/100,
                float(value[i]),0.,True) for i in range(32)]
        self.assertGreater(trainer.update(collect(0.))['optimizer_steps'],0)
        before={p:copy.deepcopy(trainer.optimizer.state[p]) for p in model.actor_parameters()}
        weights=[p.detach().clone() for p in model.actor_parameters()]
        report=trainer.update(collect(1.))
        self.assertEqual(report['optimizer_steps'],0)
        self.assertEqual(report['critic_optimizer_steps'],4)
        for p,weight in zip(model.actor_parameters(),weights):
            self.assertTrue(torch.equal(p,weight))
            for key,value in before[p].items():
                self.assertTrue(torch.equal(value,trainer.optimizer.state[p][key]))

    def test_old_checkpoint_cannot_be_resumed_as_restart(self):
        with TemporaryDirectory() as folder:
            model=LegacyModel();trainer=LegacyTrainer(model)
            path=Path(folder)/'legacy.pt';config={'source_commit':'original'}
            atomic_checkpoint(path,train_checkpoint(model,trainer,1,config,'lock',{}))
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                load_trainer(path,config,'lock')
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
