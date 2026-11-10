"""New collection/diagnostic contracts. Synthetic fixtures are not codec gains."""
from argparse import Namespace
from contextlib import ExitStack
import ast
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_v4 import UPDATE_VERSION
from agent_v4.checkpoint import load
from agent_v4.diagnostics import action_diagnostics,fixed_control_maps,spatial_credit_diagnostics
from agent_v4.session import run_frozen_jobs
from agent_v4.train import due_validation,next_batch_size
from agent_v4.state import FRAME_BYTES
from real_reward_pilot import sha256_file
from test_agent_v4 import batch


class FrozenCollection(unittest.TestCase):
    def test_four_clips_reuse_one_checkpoint_across_two_gpu_waves(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'weights';cp.write_bytes(b'frozen');seen=[]
            def runner(jobs,*args):seen.append((jobs.copy(),sha256_file(cp)))
            self.assertTrue(run_frozen_jobs(list(range(4)),None,None,cp,None,60,2,runner=runner))
            self.assertEqual([x[0] for x in seen],[[0,1],[2,3]])
            self.assertEqual(len({x[1] for x in seen}),1)

    def test_policy_mutation_stops_before_next_wave(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'weights';cp.write_bytes(b'frozen');seen=[]
            def runner(jobs,*args):seen.append(jobs);cp.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed during'):
                run_frozen_jobs(list(range(4)),None,None,cp,None,60,2,runner=runner)
            self.assertEqual(len(seen),1)

    def test_deadline_after_wave_abandons_batch(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'weights';cp.write_bytes(b'frozen');seen=[]
            with patch('agent_v4.session.time.monotonic',side_effect=[0,1000]):
                ok=run_frozen_jobs(list(range(4)),None,None,cp,None,60,2,1000,runner=lambda jobs,*args:seen.append(jobs))
            self.assertFalse(ok);self.assertEqual(seen,[[0,1]])

    def test_session_epoch_and_validation_boundaries(self):
        self.assertEqual(next_batch_size(8,10,4,20),2)
        self.assertEqual(next_batch_size(18,50,4,20),2)
        self.assertEqual(next_batch_size(324,500,4,20),1)
        p=dict(completed=[{}]*20,updates=[{}]*5,validations=[])
        self.assertEqual(due_validation(p,25,every_episodes=20),'quick')
        p['completed']=[{}]*16;p['updates']=[{}]*4
        self.assertIsNone(due_validation(p,4,every_episodes=20))


class SpatialDiagnostics(unittest.TestCase):
    def test_uniform_modes_do_not_imply_identical_probabilities(self):
        probs=np.full((12,5),.1);probs[:,2]=.6
        first={'poc':1,'action':[0]*12,'probabilities':probs.tolist()}
        second={'poc':0,'action':[-1]*6+[0]*6}
        a=action_diagnostics([first]);self.assertEqual(a['mixed_delta_frame_fraction'],0.)
        self.assertAlmostEqual(a['within_frame_policy_js'],0.)
        a=action_diagnostics([first,second]);self.assertEqual(a['mixed_delta_frame_fraction'],.5)
        controls=fixed_control_maps([first,second],2)
        self.assertEqual(controls['frame_uniform_replay'],[[0]*12,[0]*12])
        self.assertEqual(controls['uniform_minus1'],[[-1]*12,[-1]*12])

    def test_control_replay_rejects_missing_duplicate_or_illegal_actions(self):
        for rows in ([{'poc':0,'action':[0]*12}]*2,[{'poc':0,'action':[0.1]*12}],[]):
            with self.assertRaises(ValueError):fixed_control_maps(rows,2)

    def test_spatial_credit_distinguishes_global_and_local_variation(self):
        p=torch.full((4,12,5),.2);b=torch.zeros(4,12)
        a=torch.arange(4).float()[:,None].expand(-1,12)
        result=spatial_credit_diagnostics(a,b,p)
        self.assertEqual(result['advantage_within_frame_std'],0.)
        self.assertGreater(result['advantage_frame_mean_std'],0.)
        result=spatial_credit_diagnostics(a-torch.arange(12),b,p)
        self.assertGreater(result['advantage_within_frame_std'],0.)

    def test_old_pilot_checkpoint_rejected_by_update_contract(self):
        self.assertEqual(UPDATE_VERSION,'frozen-multiclip-ctu-ppo-outcome-v4')
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'old.pt'
            torch.save(dict(schema='vvenc-agent-v4/2',model_version='ctu-categorical-counterfactual-outcome-v4.2',
                state_version='causal-ctu-source-history/1',update_version='final-score-masked-ctu-ppo-outcome-aux-v3'),cp)
            with self.assertRaisesRegex(ValueError,'version'):load(cp)


class MultiClipCoordinator(unittest.TestCase):
    def exercise(self,fail_second_wave=False):
        import agent_v4.train as train
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            root=Path(td);repo=root/'repo';enc=repo/'bin/release-static/vvencapp'
            enc.parent.mkdir(parents=True);enc.write_bytes(b'encoder');dec=root/'vvdecapp';dec.write_bytes(b'decoder')
            (root/'vvdecapp_poc_audit').write_bytes(b'auditor')
            args=Namespace(repo=repo,decoder=dec,output=root/'run',input_root=root/'input',resume_archive=None,
                preset='encoder-history',reward=None,rate_mode='symmetric',rate_tolerance=0.,beta=.1,auxiliary=0.,
                frames=32,validation_frames=32,epochs=1,max_episodes=10,seed=19,workers=2,
                rollouts_per_update=4,ppo_minibatch=64,actor_lr=1e-4,val_every_updates=25,
                val_every_episodes=20,validation_spatial_controls=True,worker_timeout=60,max_hours=10.)
            splits={k:[f'{k}{i}' for i in range(n)] for k,n in [('train',65),('val',7),('test',100)]}
            waves=[]
            def jobs(jobs,repo,decoder,checkpoint,*unused):
                ckpt,model,_=load(checkpoint)
                is_train=jobs[0]['split']=='train'
                if is_train:
                    committed=json.loads((args.output/'progress.json').read_text())
                    waves.append((len(committed['completed']),sha256_file(checkpoint),len(jobs)))
                    if fail_second_wave and len(waves)==2:raise RuntimeError('second wave failed')
                for job in jobs:
                    out=job['output'];out.mkdir(parents=True)
                    src=out/'reference_8bit.yuv';src.write_bytes(bytes(FRAME_BYTES*job['frames']))
                    s=batch(job['frames']);torch.manual_seed(job['seed'])
                    greedy=job['split']=='val' and not job.get('stochastic_validation')
                    with torch.no_grad():aa,lp,v,_=model.sample(s,greedy)
                    rr=[]
                    for i in range(job['frames']):
                        rr.append(dict(poc=i,**{k:t[i].tolist() for k,t in s.items() if k!='image'},
                            action=aa[i].tolist(),log_prob=float(lp[i]),value=float(v[i]),
                            reward=.5*(i==0)+float(aa[i].float().mean())*.001,
                            done=i==job['frames']-1,trainable=is_train,
                            teacher_occupancy=[0.]*12,task_outcomes=[[0.,0.,0.]]*12))
                    rollout=out/'rollout.jsonl';rollout.write_text('\n'.join(json.dumps(r) for r in rr))
                    score=sum(r['reward'] for r in rr) if is_train else .5
                    summary=dict(split=job['split'],index=job['index'],video_id=splits[job['split']][job['index']],
                        frames=job['frames'],rate=job['rate'],start_frame=job['start'],seed=job['seed'],
                        deterministic=greedy,policy_sha256=sha256_file(checkpoint),experiment=ckpt['experiment'],
                        source_sha256=sha256_file(src),rollout_sha256=sha256_file(rollout),
                        metrics=dict(score=score,selection_score=score,f1=.5,actual_kbps=job['rate']),
                        action_histogram={'0':job['frames']*12},delta_score=score-.5,delta_f1=0.,delta_kbps=0.,vanilla={'score':.5})
                    if job.get('spatial_controls'):
                        summary['spatial_controls']={name:dict(metrics=dict(score=.99),delta_score_vs_vanilla=.49,
                            delta_f1_vs_vanilla=.3,delta_kbps_vs_vanilla=1.) for name in ('uniform_minus1','frame_uniform_replay')}
                    (out/'summary.json').write_text(json.dumps(summary))
            for target,value in [
                ('torch.cuda.is_available',lambda:True),('torch.cuda.device_count',lambda:2),
                ('agent_v4.train.ENCODER_SHA',sha256_file(enc)),('agent_v4.train.DECODER_SHA',sha256_file(dec)),
                ('agent_v4.train.read_lock_manifest',lambda x:(splits,{})),
                ('agent_v4.train.read_baseline_config',lambda x:{'lock_sha256':'lock'}),
                ('agent_v4.train.initialize_detector_assets',lambda *x:None),
                ('agent_v4.train.dataset_source',lambda *x:Path('train.mov')),
                ('agent_v4.train.video_frame_capacity',lambda x:1000),
                ('agent_v4.train.subprocess.check_output',lambda cmd,**kw:'a'*40 if 'rev-parse' in cmd else ''),
                ('agent_v4.train.run_jobs',jobs)]:stack.enter_context(patch(target,value))
            if fail_second_wave:
                with self.assertRaisesRegex(RuntimeError,'second wave failed'):train.run(args)
                p=json.loads((args.output/'progress.json').read_text())
                self.assertEqual(len(p['completed']),0);self.assertEqual(len(p['updates']),0)
                ckpt,_,_=load(args.output/p['latest_checkpoint']);self.assertEqual(ckpt['actor_optimizer']['state'],{})
                self.assertTrue((args.output/'VVC_AGENT_V4_RESUME.zip').is_file())
            else:
                train.run(args);p=json.loads((args.output/'progress.json').read_text())
                self.assertEqual([len(u['steps']) for u in p['updates']],[4,4,2])
                self.assertEqual([u['transitions'] for u in p['updates']],[128,128,64])
                self.assertEqual([x[0] for x in waves],[0,0,4,4,8])
                self.assertEqual(waves[0][1],waves[1][1]);self.assertEqual(waves[2][1],waves[3][1])
                self.assertEqual([v['episodes'] for v in p['validations']],[0,10])
                self.assertEqual(p['validations'][-1]['mean_score'],.5)
                self.assertFalse(p['validations'][-1]['spatial_controls']['uniform_minus1']['selection_eligible'])
                self.assertNotIn('best_checkpoint',p)
                self.assertEqual(p['validations'][-1]['sampled_f1_change_from_initial'],0.)
                self.assertEqual(p['validations'][-1]['spatial_learning_signal'],'NO_FRAME_REPLAY_GAIN_ESTABLISHED')
                self.assertIn('local_state',p['updates'][0]['metrics']['actor_gradient_rms'])
                self.assertIn('spatial_credit',p['updates'][0]['metrics'])
                args.resume_archive=args.output/'VVC_AGENT_V4_RESUME.zip';args.output=root/'restored';args.max_episodes=2
                train.run(args);restored=json.loads((args.output/'progress.json').read_text())
                self.assertEqual(len(restored['completed']),12)
                self.assertEqual(restored['updates'][-1]['collecting_policy_sha256'],p['latest_sha256'])

    def test_ten_episode_four_clip_commits_and_portable_resume(self):self.exercise()
    def test_second_wave_failure_preserves_preupdate_checkpoint(self):self.exercise(True)


class MeasuredControls(unittest.TestCase):
    def exercise(self,bad_stream=False,bad_metrics=False):
        import agent_v4.collect as collect
        from agent_v4.reward import RewardConfig
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);online=root/'online.266';online.write_bytes(b'original')
            rows=[dict(poc=1,action=[-1]*12),dict(poc=0,action=[-1]*12)]
            metrics={'score':.6,'f1':.7,'actual_kbps':120.}
            def encode(encoder,source,out,frames,rate,maps,threads):
                out.mkdir(parents=True);stream=out/'fixed.266'
                stream.write_bytes(b'changed' if bad_stream else b'original')
                return stream,[]
            def joined(*args):return [],dict(metrics,score=.9) if bad_metrics else metrics.copy()
            with patch.object(collect,'encode_fixed_map',side_effect=encode),\
                    patch.object(collect,'decode_detect',return_value=({},dict(verified=True))),\
                    patch.object(collect,'native_decision_rows',return_value=[{'poc':0},{'poc':1}]),\
                    patch.object(collect,'join',side_effect=joined):
                result=collect.collect_spatial_controls(Path('encoder'),Path('source'),Path('decoder'),None,
                    root,2,120,'video',{},rows,online,metrics,dict(metrics,score=.5,f1=.6),RewardConfig(),4)
            self.assertEqual(set(result),{'uniform_minus1','frame_uniform_replay'})
            for name,r in result.items():
                self.assertFalse(r['trainable']);self.assertFalse(r['selection_eligible'])
                self.assertFalse(r['equal_actual_bitrate_claim']);self.assertTrue(r['same_map_as_greedy'])
                self.assertAlmostEqual(r['delta_score_vs_vanilla'],.1)
                self.assertTrue((root/'controls'/name/'summary.json').is_file())

    def test_controls_record_measured_outcomes_without_training(self):self.exercise()
    def test_identical_nonzero_map_must_reproduce_live_stream(self):
        with self.assertRaisesRegex(ValueError,'replay differs'):self.exercise(bad_stream=True)
    def test_identical_stream_cannot_change_detector_metrics(self):
        with self.assertRaisesRegex(ValueError,'different task metrics'):self.exercise(bad_metrics=True)


class NotebookContracts(unittest.TestCase):
    def test_epoch1_and_resume_keep_the_frozen_experiment_configuration(self):
        nb=Path(__file__).resolve().parents[2]/'notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb'
        cells=json.loads(nb.read_text(encoding='utf-8'))['cells']
        config={};exec(cells[1]['source'],config)
        self.assertEqual(config['MAX_ADDITIONAL_EPISODES'],325)
        self.assertEqual(config['NEW_EPOCHS'],1)
        commands=[]
        ns=dict(config,Path=Path,sys=sys,time=types.SimpleNamespace(monotonic=lambda:0),SESSION_START=0,
            EXP=None,REPO=Path('/repo'),DEC=Path('/dec'),RUN=Path('/nonexisting/run'),RESUME_ARCHIVE=None,
            run=lambda cmd,**kw:commands.append(cmd) or 0,export_session=lambda ctx:None,record_session_failure=lambda ctx:None)
        exec(cells[6]['source'],ns)
        previous=json.loads(json.dumps(ns['cfg']))
        ns.update(EXP=previous,MAX_ADDITIONAL_EPISODES=325,RESUME_ARCHIVE=Path('/input/resume.zip'),
            NEW_EPOCHS=3,
            NEW_ROLLOUTS_PER_UPDATE=8,NEW_PPO_MINIBATCH=128,NEW_ACTOR_LR=.2,
            NEW_VAL_EVERY_EPISODES=99,NEW_SPATIAL_CONTROLS=False)
        exec(cells[6]['source'],ns)
        for i,cmd in enumerate(commands):
            for option,value in [('--rollouts-per-update',4),('--ppo-minibatch',64),('--actor-lr',1e-4),('--val-every-episodes',20)]:
                self.assertEqual(cmd[cmd.index(option)+1],value)
            self.assertIn('--validation-spatial-controls',cmd)
            self.assertEqual(cmd[cmd.index('--max-episodes')+1],325)
            self.assertEqual(cmd[cmd.index('--epochs')+1],1)
        self.assertNotIn('--resume-archive',commands[0]);self.assertIn('--resume-archive',commands[1])

    def test_update_and_collection_guard_match_source(self):
        nb=Path(__file__).resolve().parents[2]/'notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb'
        cells=json.loads(nb.read_text(encoding='utf-8'))['cells']
        for c in cells:
            if c['cell_type']=='code':compile(c['source'],'generated-cell','exec')
        tree=ast.parse(cells[2]['source'])
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.Assert) and any(
            key in ast.unparse(n) for key in ('collection_version','update_version'))]
        self.assertEqual(len(guards),2)
        for exp,ok in [(dict(collection_version='frozen-multiclip-waves-v1',update_version=UPDATE_VERSION),True),
                       (dict(update_version='final-score-masked-ctu-ppo-outcome-aux-v3'),False)]:
            for guard in guards:
                if ok:exec(compile(ast.Module(body=[guard],type_ignores=[]),'guard','exec'),dict(EXP=exp))
                else:
                    with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[guard],type_ignores=[]),'guard','exec'),dict(EXP=exp))


if __name__=='__main__':unittest.main()
