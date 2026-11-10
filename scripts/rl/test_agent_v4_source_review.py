"""Review regressions: candidate provenance, incomplete evaluation, graph isolation."""
from contextlib import ExitStack
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from agent_v4.checkpoint import save,load,recorded_episode_count
from agent_v4.codec import runtime_profile
from agent_v4.model import Agent,ModelConfig,PPOConfig,Trainer
from agent_v4.reward import RewardConfig
from agent_v4.train import validate_pending
from test_agent_v4 import batch


class EpisodeProvenance(unittest.TestCase):
    def test_launcher_rejects_resume_without_episode_metadata_contract(self):
        nb=Path(__file__).resolve().parents[2]/'notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb'
        setup=json.loads(nb.read_text(encoding='utf-8'))['cells'][2]['source']
        guard=next(n for n in ast.walk(ast.parse(setup)) if isinstance(n,ast.Assert) and
                   'checkpoint_metadata_version' in ast.unparse(n))
        code=compile(ast.Module(body=[guard],type_ignores=[]),'resume-guard','exec')
        exec(code,dict(EXP={'checkpoint_metadata_version':'completed-episodes-v1'}))
        with self.assertRaises(AssertionError):exec(code,dict(EXP={}))
        epoch_guard=next(n for n in ast.walk(ast.parse(setup)) if isinstance(n,ast.Assert) and
            any(isinstance(t,ast.Constant) and t.value=='epochs' for t in ast.walk(n)))
        code=compile(ast.Module(body=[epoch_guard],type_ignores=[]),'epoch-guard','exec')
        exec(code,dict(EXP={'epochs':1}))
        with self.assertRaises(AssertionError):exec(code,dict(EXP={'epochs':3}))

    def make_candidate(self,root):
        model=Agent(ModelConfig(content=False));trainer=Trainer(model,PPOConfig(auxiliary=0.))
        exp=dict(model_config=asdict(model.config),ppo_config=asdict(trainer.config),reward=asdict(RewardConfig()),
            source_commit='a'*40,codec_profile=runtime_profile(),encoder_threads=4,decoder_threads=4,
            seed=19,validation_frames=32,validation_spatial_controls=True,
            checkpoint_metadata_version='completed-episodes-v1')
        cp=root/'candidate.pt';save(cp,model,trainer,exp,3,completed_episodes=10)
        return cp

    def context(self,root,cp,out):
        stack=ExitStack()
        for target,value in [('torch.cuda.is_available',lambda:True),('torch.cuda.device_count',lambda:2),
            ('agent_v4_validate.verify_source',lambda *args:None),
            ('agent_v4_validate.read_baseline_config',lambda *args:{}),
            ('agent_v4_validate.initialize_detector_assets',lambda *args:None)]:stack.enter_context(patch(target,value))
        stack.enter_context(patch.object(sys,'argv',['agent_v4_validate.py','--repo',str(root/'repo'),
            '--decoder',str(root/'decoder'),'--checkpoint',str(cp),'--output',str(out),'--quick']))
        return stack

    def test_episode_count_survives_checkpoint_without_update_count_inference(self):
        with tempfile.TemporaryDirectory() as td:
            cp=self.make_candidate(Path(td));ckpt,_,_=load(cp)
            self.assertEqual(ckpt['iteration'],3);self.assertEqual(recorded_episode_count(ckpt),10)
            for value in (None,-1,True,2):
                with self.assertRaises(ValueError):recorded_episode_count(dict(ckpt,completed_episodes=value))

    def test_trained_standalone_candidate_is_not_mislabeled_episode_zero(self):
        import agent_v4_validate
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp=self.make_candidate(root);out=root/'val';seen=[]
            def capture(jobs,*args):
                seen.extend(jobs);raise RuntimeError('captured jobs before native execution')
            with self.context(root,cp,out),patch('agent_v4.train.run_jobs',side_effect=capture):
                with self.assertRaisesRegex(RuntimeError,'captured jobs'):agent_v4_validate.main()
            p=json.loads((out/'progress.json').read_text());pending=p['pending_validation']
            self.assertEqual(pending['episodes'],10);self.assertEqual(pending['checkpoint_iteration'],3)
            self.assertIn('000010',pending['output'])
            self.assertTrue(all(j['spatial_controls'] and not j['zero_gate'] for j in seen))

    def test_incomplete_standalone_validation_returns_nonzero(self):
        import agent_v4_validate
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp=self.make_candidate(root);out=root/'val'
            with self.context(root,cp,out),patch('agent_v4_validate.validate_pending',return_value=False):
                self.assertEqual(agent_v4_validate.main(),2)
            self.assertTrue((out/'VVC_AGENT_V4_RESULTS.zip').is_file())
            self.assertIn('pending_validation',json.loads((out/'progress.json').read_text()))

    def test_pending_checkpoint_corruption_fails_before_collectors(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp=root/'weights';cp.write_bytes(b'changed')
            args=SimpleNamespace(output=root)
            progress={'pending_validation':{'checkpoint':'weights','sha256':'bad'}}
            with patch('agent_v4.train.run_jobs') as jobs:
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):validate_pending(args,progress,{},float('inf'))
                jobs.assert_not_called()


class CriticOnlyGraph(unittest.TestCase):
    def test_value_executes_no_actor_and_backpropagates_only_through_critic(self):
        for config in (ModelConfig(content=False),ModelConfig(False,False,False)):
            model=Agent(config);s=batch(4)
            with patch.object(model,'policy_features',side_effect=AssertionError('Actor must not run')):
                model.value(s).square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.actor_parameters()))
            self.assertTrue(any(p.grad is not None for p in model.critic_parameters()))

    def test_one_update_does_not_repeat_actor_inside_critic_forward(self):
        torch.manual_seed(61)
        model=Agent(ModelConfig(content=False));trainer=Trainer(model,PPOConfig(epochs=1,minibatch=4,auxiliary=0.))
        s=batch(4)
        with torch.no_grad():actions,lp,values,_=model.sample(s)
        with patch.object(model,'forward',wraps=model.forward) as joint:
            trainer.update(s,actions,lp,values,torch.tensor([.1,.2,.3,.4]),torch.tensor([False]*3+[True]),torch.zeros(4,12))
        self.assertEqual(joint.call_count,1)  # parity only; update uses the critic-only graph


if __name__=='__main__':unittest.main()
