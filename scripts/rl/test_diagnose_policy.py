import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch

import diagnose_policy as d
from full_train import atomic_checkpoint,train_checkpoint
from ppo_agent_legacy import PPOActorCritic,PPOTrainer
from reward_metrics import Detection,calculate_frame_reward


class DiagnosticTests(unittest.TestCase):
    def test_quantized_probabilities_match_sampling(self):
        torch.manual_seed(42)
        mean=torch.tensor([[0.,-.7,1.2]]);std=torch.full_like(mean,.5)
        p=d.action_distribution(mean,std)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones_like(mean)))
        draws=torch.distributions.Normal(mean,std).sample((100000,))
        actions=(2*draws.tanh()).round().long()
        for i in range(3):
            empirical=torch.stack([(actions[:,0,i]==k).float().mean() for k in range(-2,3)])
            self.assertTrue(torch.allclose(empirical,p[0,i],atol=.006))

    def test_saturation_reduces_discrete_entropy_despite_same_gaussian_std(self):
        def entropy(mu):
            p=d.action_distribution(torch.tensor([mu]),torch.tensor([.5]))
            return float(-(p*p.clamp_min(1e-30).log()).sum())
        self.assertGreater(entropy(0.),entropy(-4.)*100)
        p=d.action_distribution(torch.tensor([-4.]),torch.tensor([.5]))
        self.assertGreater(float(p[0,0]),.999)

    def test_view_reads_zip_and_expanded_directory_and_rejects_traversal(self):
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as z:z.writestr('a.json','{"x":1}')
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'a.json').write_text('{"x":1}')
            for view in (d.View(raw.getvalue()),d.View(root)):
                self.assertEqual(view.json('a.json'),{'x':1})
                with self.assertRaises(ValueError):view.read('../secret')
                view.close()

    def test_wrong_export_metadata_is_rejected(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'export.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr(d.MANIFEST,json.dumps({'completed_episodes':20}))
            with self.assertRaises(ValueError):d.verified_bundle(p,68)

    def test_validation_only_probe_plan(self):
        self.assertEqual(d.probe_plan('0,1','80,120'),([0,1],[80,120]))
        for i,r in [('7','80'),('-1','80'),('0,0','80'),('0','75'),('0','80,80')]:
            with self.assertRaises(ValueError):d.probe_plan(i,r)

    def test_fixed_encoder_settings_and_environment(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);calls=[]
            with mock.patch.dict('os.environ',{'VVENC_RL_ACTION_DIR':'stale','VVENC_RL_TRACE':'1'}),\
                 mock.patch('real_reward_pilot.run_cmd',side_effect=lambda cmd,log,**kw:calls.append((cmd,kw))):
                d.fixed_encode(root/'enc',root/'ref',root/'fixed',32,80,-2,4)
            cmd,kw=calls[0]
            self.assertNotIn('VVENC_RL_ACTION_DIR',kw['env']);self.assertNotIn('VVENC_RL_TRACE',kw['env'])
            self.assertEqual(cmd[cmd.index('--qpa')+1],'1')
            self.assertEqual(cmd[cmd.index('--additional')+1],'GOPSize=32:LookAhead=1')
            rows=(root/'fixed/fixed_map.txt').read_text().splitlines()
            self.assertEqual(len(rows),32)
            self.assertEqual(rows[-1].split(),['31']+['-2']*12)
            with self.assertRaises(ValueError):d.fixed_encode(root/'enc',root/'ref',root/'bad',32,80,-3,4)

    def test_probe_reward_uses_policy_order_not_poc(self):
        with TemporaryDirectory() as folder:
            out=Path(folder);n=32;order=[1,0]+list(range(2,n))
            (out/'states.txt').write_text(''.join(f'STATE {p}\n' for p in order))
            stream=out/'b.266';stream.write_bytes(b'x'*3300)
            source={i:[Detection(0,0,10,10,.9)] for i in range(n)}
            bits={i:1600 if i==0 else 800 for i in range(n)}
            au=[SimpleNamespace(poc=i,au_bits=bits[i]) for i in range(n)]
            def run(cmd,*a,**kw):Path(cmd[cmd.index('-o')+1]).write_bytes(b'decoded')
            def convert(a,b,n):b.write_bytes(b'rec8')
            with mock.patch('real_reward_pilot.run_cmd',side_effect=run),\
                 mock.patch('real_reward_pilot.convert_10bit_to_8bit',side_effect=convert),\
                 mock.patch('real_reward_pilot.detect_all'),\
                 mock.patch('decoder_alignment.verify_decode_mapping',return_value={'decoded_sha256':'decodedhash'}),\
                 mock.patch('reward_join.read_detections',return_value=source),\
                 mock.patch('au_rate_audit.read_au_bits',return_value=au):
                r=d.probe_metrics(out/'dec',None,stream,out/'refdet','video',n,80,out,4)
            cumulative=0;expected=[]
            for step,poc in enumerate(order,1):
                cumulative+=bits[poc]
                expected.append(calculate_frame_reward(video_id='video',poc=poc,encoded_step=step,
                    pseudo_gt=source[poc],reconstructed=source[poc],actual_frame_bits=bits[poc],
                    cumulative_bits=cumulative,frame_rate=30,target_bps=80000,beta=.1).reward)
            self.assertAlmostEqual(r['training_style_reward'],float(np.mean(expected)))
            self.assertEqual(r['f1'],1.)
            self.assertFalse((out/'decoded_10bit.yuv').exists())

    def test_actual_replay_preserves_checkpoint_and_episode_boundaries(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=PPOActorCritic();trainer=PPOTrainer(model,epochs=2,minibatch_size=8)
        cfg={'source_commit':d.OFFICIAL_COMMIT}
        with TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'input.pt'
            atomic_checkpoint(checkpoint,train_checkpoint(model,trainer,0,cfg,d.LOCK_SHA,{}))
            before=checkpoint.read_bytes();rows=[]
            for episode in range(2):
                sp=torch.rand(8,3,3,4);gl=torch.rand(8,8)
                with torch.no_grad():raw,action,lp,value=model.sample(sp,gl)
                for i in range(8):
                    rows.append(dict(spatial=sp[i].tolist(),global_=gl[i].tolist(),raw_action=raw[i].tolist(),
                        discrete_action=action[i].tolist(),log_prob=float(lp[i]),value=float(value[i]),
                        reward=.3+i*.01,next_value=float(value[i+1]) if i<7 else 0.,done=i==7))
            for row in rows:row['global']=row.pop('global_')
            result=d.replay_update(checkpoint,cfg,rows,99,{'optimizer_steps':4})
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertTrue(result['checkpoint_bytes_unchanged'])
            self.assertEqual(result['parity']['episodes'],2)
            self.assertEqual(len(result['steps']),int(result['metrics']['optimizer_steps']))
            self.assertGreater(len(result['steps']),0)
            self.assertEqual(list(Path(folder).iterdir()),[checkpoint])

    def test_offline_refuses_existing_directory(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):d.offline(Path('missing'),Path('missing'),Path(folder))


if __name__=='__main__':unittest.main()
