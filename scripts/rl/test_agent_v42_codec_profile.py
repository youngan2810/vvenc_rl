"""Runtime protocol regressions. No codec/GPU execution or quality claims."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from agent_v4 import codec


class RuntimeProfileTests(unittest.TestCase):
    def test_profile_rejects_old_or_tampered_experiments(self):
        exp={'codec_profile':codec.runtime_profile(),'encoder_threads':4,'decoder_threads':4}
        codec.require_runtime_profile(exp)
        for key,value in (('codec_profile',None),('encoder_threads',1),('decoder_threads',1)):
            changed=dict(exp);changed[key]=value
            with self.assertRaisesRegex(ValueError,'NEW fpp1'):codec.require_runtime_profile(changed)
        changed=dict(exp);changed['codec_profile']=dict(exp['codec_profile'],max_parallel_frames=4)
        with self.assertRaises(ValueError):codec.require_runtime_profile(changed)
        self.assertFalse(exp['codec_profile']['historical_500_pair_baseline_compatible'])

    def test_effective_header_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            log=Path(td)/'encoder.log'
            for header,ok in (('Threads : 4  (parallel frames: 1)',True),
                              ('Threads : 4 (parallel frames: 4)',False),
                              ('Threads : 1 (parallel frames: 1)',False),('',False)):
                log.write_text(header)
                if ok:codec.verify_effective_profile(log)
                else:
                    with self.assertRaises(ValueError):codec.verify_effective_profile(log)

    def test_common_command_preserves_explicit_profile(self):
        cmd=codec.encoder_command('enc','in.yuv','out.266',256,120)
        self.assertEqual(cmd[cmd.index('--additional')+1],'GOPSize=32:LookAhead=1:MaxParallelFrames=1')
        self.assertEqual(cmd[cmd.index('--threads')+1],'4')
        self.assertEqual(cmd[cmd.index('--bitrate')+1],'120k')
        with self.assertRaises(ValueError):codec.encoder_command('e','i','o',256,120,1)

    def test_vanilla_and_fixed_use_same_command_and_clean_environment(self):
        calls=[]
        def native(cmd,log,**kwargs):
            calls.append((cmd,kwargs['env']))
            log.write_text('Threads : 4 (parallel frames: 1)')
        with tempfile.TemporaryDirectory() as td, patch.object(codec,'run_cmd',side_effect=native), \
                patch.object(codec,'read_au_bits',return_value=[]), \
                patch.dict(codec.os.environ,{'VVENC_RL_ACTION_DIR':'stale','VVENC_RL_QP_MAP':'stale'}):
            root=Path(td)
            codec.vanilla_encode('enc',root/'src',root/'base',32,120)
            codec.encode_fixed_map('enc',root/'src',root/'fixed',32,120,[[0]*12 for _ in range(32)])
            first,second=calls
            self.assertEqual(first[0][:-1],second[0][:-1])
            self.assertNotIn('VVENC_RL_ACTION_DIR',first[1]);self.assertNotIn('VVENC_RL_QP_MAP',first[1])
            self.assertNotIn('VVENC_RL_ACTION_DIR',second[1]);self.assertIn('VVENC_RL_QP_MAP',second[1])

    def test_vanilla_fails_when_native_ignores_setting(self):
        def native(cmd,log,**kwargs):log.write_text('Threads : 4 (parallel frames: 4)')
        with tempfile.TemporaryDirectory() as td, patch.object(codec,'run_cmd',side_effect=native):
            with self.assertRaisesRegex(ValueError,'does not confirm'):
                codec.vanilla_encode('enc',Path('src'),Path(td)/'base',32,120)

    def test_live_bridge_uses_same_native_command_and_effective_gate(self):
        class Bridge:
            def __init__(self,cmd,**kwargs):
                Path(cmd[cmd.index('--ready-file')+1]).write_text('ready')
                Path(cmd[cmd.index('--trajectory-file')+1]).write_text(json.dumps({'poc':0})+'\n')
            def poll(self):return 0
            def wait(self,**kwargs):return 0
        def native(cmd,log,**kwargs):
            self.assertEqual(cmd,codec.encoder_command('enc',Path('src'),log.parent/'ppo.266',1,120))
            self.assertIn('VVENC_RL_ACTION_DIR',kwargs['env'])
            log.write_text(header)
        with tempfile.TemporaryDirectory() as td, patch.object(codec.subprocess,'Popen',Bridge), \
                patch.object(codec,'run_cmd',side_effect=native),patch.object(codec,'read_au_bits',return_value=[]):
            for i,header in enumerate(('Threads : 4 (parallel frames: 1)','Threads : 4 (parallel frames: 4)')):
                args=(Path('repo'),'enc',Path('src'),Path(td)/str(i),1,Path('checkpoint'),120)
                if i==0:codec.encode(*args)
                else:
                    with self.assertRaisesRegex(ValueError,'does not confirm'):codec.encode(*args)


if __name__=='__main__':unittest.main()
