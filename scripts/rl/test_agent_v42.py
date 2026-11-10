"""Learning-contract regressions: these are not native codec quality tests."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from agent_v4 import SCHEMA, MODEL_VERSION, UPDATE_VERSION
from agent_v4.checkpoint import load
from agent_v4.codec import require_zero_identity, encode_fixed_map
from agent_v4.learning import episode_scores, counterfactual_advantages, merge_actor_gradients
from agent_v4.model import Agent, ModelConfig, PPOConfig, Trainer
from agent_v4.reward import local_task_outcomes
from reward_metrics import Detection, match_one_to_one
from test_agent_v4 import batch

torch.set_num_threads(1)


class LearningContracts(unittest.TestCase):
    def test_final_scores_not_prefix_tails_and_never_cross_clips(self):
        r=torch.tensor([.7,-.1,.1,.2])
        done=torch.tensor([False,True,False,True])
        torch.testing.assert_close(episode_scores(r,done),torch.tensor([.6,.6,.3,.3]))
        with self.assertRaises(ValueError):
            episode_scores(r,torch.zeros(4,dtype=torch.bool))

    def test_counterfactual_gradient_equals_exact_enumeration(self):
        # Arbitrarily WRONG Q is still a valid baseline if frozen and masked.
        logits=torch.tensor([[.3,-.1,.2,0.,-.2],[-.2,.2,0.,.1,.4]],dtype=torch.float64,requires_grad=True)
        p=logits.softmax(-1)
        expected=logits.sum()*0
        surrogate=logits.sum()*0
        for a,b in itertools.product(range(5),repeat=2):
            joint=p[0,a]*p[1,b]
            outcome=float((a-2)*(b-2)+.3*a-.2*b)
            expected=expected+joint*outcome
            probs=torch.zeros(1,12,5,dtype=torch.float64)
            probs[0,:2]=p.detach();probs[0,2:,2]=1
            q=torch.zeros_like(probs)
            q[0,0]=torch.arange(5,dtype=torch.float64)*17+b*13
            q[0,1]=torch.arange(5,dtype=torch.float64)*-11+a*19
            adv,_=counterfactual_advantages(torch.tensor([outcome],dtype=torch.float64),probs,q)
            surrogate=surrogate+joint.detach()*(p[0,a].log()*adv[0,0]+p[1,b].log()*adv[0,1])
        actual=torch.autograd.grad(surrogate,logits,retain_graph=True)[0]
        truth=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(actual,truth,atol=1e-12,rtol=1e-12)

    def test_counterfactual_masks_own_action_before_mixing(self):
        for config in (ModelConfig(content=False),ModelConfig(False,False,False)):
            m=Agent(config)
            torch.nn.init.normal_(m.counterfactual_head[-1].weight)
            s=batch(2);a=torch.zeros(2,12,dtype=torch.long)
            q=m.counterfactual_values(s,a)
            for i in (0,5,11):
                other=a.clone();other[:,i]=2
                changed=m.counterfactual_values(s,other)
                torch.testing.assert_close(q[:,i],changed[:,i],atol=0,rtol=0)
                self.assertGreater(float((q[:,(i+1)%12]-changed[:,(i+1)%12]).abs().sum()),0)

    def test_action_conditioned_outcome_and_actor_content_gradient(self):
        m=Agent();s=batch(2);s['image'].random_(16,236)
        _,tokens=m.policy_features(s)
        a=torch.zeros(2,12,dtype=torch.long)
        first=m.task_outcomes(tokens,a)
        second=m.task_outcomes(tokens,a+1)
        self.assertGreater(float((first-second).abs().sum()),0)
        first.square().mean().backward()
        self.assertGreater(float(m.actor.image[0].weight.grad.abs().sum()),0)
        self.assertIsNone(m.critic.image[0].weight.grad)

    def test_auxiliary_conflict_projection_and_budget(self):
        p=torch.nn.Parameter(torch.zeros(2))
        merged,d=merge_actor_gradients([torch.tensor([1.,0.])],[torch.tensor([-10.,10.])],[p],.5)
        torch.testing.assert_close(merged[0],torch.tensor([1.,.5]))
        self.assertEqual(d['auxiliary_grad_conflict'],1.)
        merged,_=merge_actor_gradients([None],[torch.ones(2)],[p],.5)
        self.assertEqual(float(merged[0].abs().sum()),0.)

    def test_measured_labels_required_and_paired_baseline_constant_per_clip(self):
        m=Agent(ModelConfig(content=False));s=batch(4)
        t=Trainer(m,PPOConfig(epochs=1,paired_baseline=True))
        with torch.no_grad():a,lp,v,_=m.sample(s)
        r=torch.tensor([.2,-.1,.2,.1]);d=torch.tensor([False,True,False,True])
        occ=torch.zeros(4,12)
        with self.assertRaisesRegex(ValueError,'measured'):
            t.update(s,a,lp,v,r,d,occ)
        with self.assertRaisesRegex(ValueError,'vanilla'):
            t.update(s,a,lp,v,r,d,occ,torch.zeros(4,12,3))
        with self.assertRaisesRegex(ValueError,'constant'):
            t.update(s,a,lp,v,r,d,occ,torch.zeros(4,12,3),torch.arange(4).float())
        metrics=t.update(s,a,lp,v,r,d,occ,torch.rand(4,12,3),torch.full((4,),.1))
        self.assertEqual(metrics['credit_assignment'],'counterfactual-final')
        self.assertEqual(metrics['counterfactual_baseline_std'],0.)  # frozen zero-initialized critic
        self.assertGreater(float(m.counterfactual_head[-1].weight.abs().sum()),0)
        self.assertLessEqual(metrics['exact_final_kl'],.03)

    def test_shared_final_ablation_and_legacy_control_remain_explicit(self):
        for credit in ('shared-final','joint-gae'):
            m=Agent(ModelConfig(content=False));s=batch(4)
            t=Trainer(m,PPOConfig(epochs=1,auxiliary=0.,credit_assignment=credit))
            with torch.no_grad():a,lp,v,_=m.sample(s)
            mm=t.update(s,a,lp,v,torch.tensor([.1,.2,.3,-.1]),torch.tensor([False]*3+[True]),torch.zeros(4,12))
            self.assertEqual(mm['credit_assignment'],credit)
        with self.assertRaises(ValueError):
            PPOConfig(gamma=.99)

    def test_learns_context_specific_nonzero_actions_from_real_sampled_rewards(self):
        # Known two-context cooperative bandit: useful greedy actions differ
        # by state. This tests learning behavior, not codec gain or only shapes.
        torch.manual_seed(319)
        m=Agent(ModelConfig(content=False))
        t=Trainer(m,PPOConfig(epochs=2,minibatch=32,actor_lr=.001,critic_lr=.001,auxiliary=0.))
        s=batch(64);s['local'].zero_();s['history'].zero_()
        s['global'][:32,0]=-1;s['global'][32:,0]=1
        desired=torch.cat((torch.full((32,12),-1),torch.full((32,12),1)))
        with torch.no_grad():
            initial=float((m.sample(s,True)[0]==desired).float().mean())
        for _ in range(50):
            with torch.no_grad():a,lp,v,_=m.sample(s)
            reward=(a==desired).float().mean(1)
            t.update(s,a,lp,v,reward,torch.ones(64,dtype=torch.bool),torch.zeros(64,12))
        with torch.no_grad():
            final=float((m.sample(s,True)[0]==desired).float().mean())
        self.assertEqual(initial,0.)
        self.assertGreaterEqual(final,.95)

    def test_v41_checkpoint_rejected_before_weights_or_optimizer_load(self):
        self.assertEqual(SCHEMA,'vvenc-agent-v4/2')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'v41.pt'
            torch.save({'schema':'vvenc-agent-v4/1','model_version':'ctu-categorical-content-history-v4.1',
                        'update_version':'categorical-clip-aux-fullbatch-kl-v2'},path)
            with self.assertRaisesRegex(ValueError,'version'):
                load(path)


class OutcomeContracts(unittest.TestCase):
    def test_cross_ctu_boxes_and_duplicate_predictions_conserve_global_counts(self):
        ref=[Detection(110,110,270,270),Detection(384,256,480,320)]
        dec=[Detection(110,110,270,270,.9),Detection(110,110,270,270,.8),Detection(0,0,20,20,.7)]
        out=torch.tensor(local_task_outcomes(ref,dec))
        c=match_one_to_one(ref,dec)
        torch.testing.assert_close(out.sum(0),torch.tensor([c.tp,c.fp,c.fn]).float())
        self.assertGreater(int((out[:,0]>0).sum()),1)
        self.assertEqual(float(out[11,2]),1.)

    def test_empty_outcomes_and_out_of_frame_boxes(self):
        self.assertEqual(local_task_outcomes([],[]),[[0.,0.,0.]]*12)
        with self.assertRaises(ValueError):
            local_task_outcomes([Detection(500,500,510,510)],[])

    def test_every_allzero_pair_fails_closed_for_stream_or_detector_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.266';b=root/'b.266';a.write_bytes(b'a');b.write_bytes(b'b')
            rows=[{'action':[0]*12}];m={'tp':1,'score':.5}
            with self.assertRaisesRegex(ValueError,'bitstream'):
                require_zero_identity(a,b,rows,m,m,root)
            self.assertTrue((root/'zero_identity_failure.json').exists())
            b.write_bytes(b'a')
            with self.assertRaisesRegex(ValueError,'task metrics'):
                require_zero_identity(a,b,rows,m,{**m,'tp':2},root)
            self.assertTrue(require_zero_identity(a,b,rows,m,{**m,'score':.5+1e-16},root))
            self.assertFalse(require_zero_identity(a,b,[{'action':[1]*12}],m,m,root))

    def test_fixed_replay_requires_legal_complete_maps(self):
        with self.assertRaises(ValueError):
            encode_fixed_map(Path('enc'),Path('src'),Path('out'),2,120,[[0]*12])
        with self.assertRaises(ValueError):
            encode_fixed_map(Path('enc'),Path('src'),Path('out'),1,120,[[.1]*12])


if __name__ == '__main__':
    unittest.main()
