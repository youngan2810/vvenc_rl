"""Regression tests for measured targets, interruption recovery and failure evidence."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from agent_v4.learning import measured_episode_scores
from agent_v4.model import Agent,ModelConfig,PPOConfig,Trainer
from agent_v4.notebook_support import export_session,record_session_failure
from agent_v4.train import due_validation,stage_validation
from test_agent_v4 import batch


class MeasuredTargets(unittest.TestCase):
    def test_identical_pair_cancels_before_float32_cast(self):
        torch.manual_seed(0)
        prefix=torch.rand(256,dtype=torch.float64);prefix[-1]=.884627281163
        rewards=torch.diff(torch.cat((torch.zeros(1,dtype=torch.float64),prefix))).float()
        dones=torch.zeros(256,dtype=torch.bool);dones[-1]=True
        measured=prefix[-1].expand(256)
        self.assertNotEqual(float(rewards.sum()-measured[0].float()),0.)
        actual=(measured_episode_scores(rewards,dones,measured)-measured).float()
        self.assertTrue(torch.equal(actual,torch.zeros_like(actual)))

    def test_measured_score_rejects_wrong_clip_or_inconsistent_score(self):
        rewards=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
        dones=torch.tensor([False,True,False,True])
        expected=torch.tensor([.3,.3,.7,.7],dtype=torch.float64)
        torch.testing.assert_close(measured_episode_scores(rewards,dones,expected),expected)
        for changed in (torch.tensor([.3,.4,.7,.7]),torch.tensor([.9,.9,.7,.7])):
            with self.assertRaises(ValueError):measured_episode_scores(rewards,dones,changed)

    def test_trainer_uses_measured_zero_delta_without_reward_roundoff_gradient(self):
        torch.manual_seed(51)
        model=Agent(ModelConfig(content=False))
        trainer=Trainer(model,PPOConfig(epochs=1,auxiliary=0.,entropy=0.,paired_baseline=True))
        states=batch(4)
        with torch.no_grad():actions,lp,values,_=model.sample(states)
        before=[p.detach().clone() for p in model.actor_parameters()]
        rewards=torch.tensor([.1,.2,-.1,.4],dtype=torch.float64)
        scores=torch.full((4,),float(rewards.sum()),dtype=torch.float64)
        result=trainer.update(states,actions,lp,values,rewards,torch.tensor([False]*3+[True]),
            torch.zeros(4,12),baseline_scores=scores,final_scores=scores)
        self.assertEqual(result['observed_target_mean'],0.)
        self.assertEqual(result['advantage_std'],0.)
        for a,b in zip(before,model.actor_parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)


class ValidationRecovery(unittest.TestCase):
    def progress(self,n,updates):
        return dict(completed=[{}]*n,updates=[{}]*updates,validations=[],
                    latest_checkpoint='checkpoints/latest.pt',latest_sha256='abc')

    def test_recovers_missed_periodic_quick_validation(self):
        p=self.progress(20,10)
        self.assertEqual(due_validation(p,10),'quick')
        stage_validation(p,'quick')
        self.assertEqual(p['pending_validation']['episodes'],20)
        self.assertEqual(p['pending_validation']['checkpoint'],p['latest_checkpoint'])
        self.assertIsNone(due_validation(p,10))
        del p['pending_validation'];p['validations']=[dict(episodes=20,kind='quick')]
        self.assertIsNone(due_validation(p,10))

    def test_initial_epoch_and_session_end_schedule(self):
        self.assertEqual(due_validation(self.progress(0,0),10),'quick')
        self.assertEqual(due_validation(self.progress(325,163),10),'full')
        self.assertIsNone(due_validation(self.progress(2,1),10))
        self.assertEqual(due_validation(self.progress(2,1),10,session_end=True),'quick')


class NotebookEvidence(unittest.TestCase):
    def context(self,root):
        logs=root/'logs';logs.mkdir()
        return dict(WORK=root,RUN=root/'run',LOGS=logs,PREFLIGHT=None,HEADROOM=None,
                    SOURCE_COMMIT='test-source',TRAIN_STARTED=False,TRAIN_RC=None,
                    SESSION_STAGE='setup',record_session_failure=record_session_failure)

    def test_failure_before_progress_exports_logs_and_no_stale_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ctx=self.context(root)
            (ctx['LOGS']/'failed.log').write_text('synthetic setup failure')
            ctx['RUN'].mkdir();(ctx['RUN']/'VVC_AGENT_V4_RESUME.zip').write_bytes(b'stale')
            with contextlib.redirect_stdout(io.StringIO()):dest=export_session(ctx)
            with zipfile.ZipFile(dest) as z:
                self.assertIn('setup_logs/failed.log',z.namelist())
                self.assertNotIn('VVC_AGENT_V4_RESUME.zip',z.namelist())
                status=json.loads(z.read('session_status.json'))
                self.assertEqual(status['completed'],0);self.assertFalse(status['training_started'])

    def test_failure_evidence_excludes_raw_yuv_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ctx=self.context(root);ctx['HEADROOM']=root/'headroom';ctx['HEADROOM'].mkdir()
            for name in ('headroom.json','encoder.log','raw.yuv','weights.pt'):
                (ctx['HEADROOM']/name).write_text('fixture')
            with contextlib.redirect_stdout(io.StringIO()):dest=export_session(ctx)
            with zipfile.ZipFile(dest) as z:
                self.assertIn('train_headroom/encoder.log',z.namelist())
                self.assertFalse(any(n.endswith(('.yuv','.pt')) for n in z.namelist()))

    def test_generated_cuda_and_build_failures_export_before_reraising(self):
        nb=Path(__file__).resolve().parents[2]/'notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb'
        cells=json.loads(nb.read_text(encoding='utf-8'))['cells']
        for index in (2,4):
            tree=ast.parse(cells[index]['source'])
            guarded=next(n for n in tree.body if isinstance(n,ast.Try))
            with tempfile.TemporaryDirectory() as td:
                ctx=self.context(Path(td))
                def fail(*args,**kwargs):raise RuntimeError('synthetic build failed')
                ctx.update(run=fail,sys=__import__('sys'),REPO=Path(td)/'repo')
                with patch('torch.cuda.is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises((AssertionError,RuntimeError)):
                        exec(compile(ast.Module(body=[guarded],type_ignores=[]),'generated-cell','exec'),ctx)
                with zipfile.ZipFile(Path(td)/'VVC_AGENT_V4_KAGGLE_SESSION.zip') as z:
                    self.assertIn('setup_logs/failure_traceback.log',z.namelist())
                    self.assertEqual(json.loads(z.read('session_status.json'))['completed'],0)


if __name__=='__main__':unittest.main()
