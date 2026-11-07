"""Exercise an actual bridge process while its STATE file is being appended."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest import TestCase, main


class BridgeIOTests(TestCase):
    def run_bridge(self,folder,kind):
        root=Path(folder);root.mkdir(exist_ok=True);states=root/'states.txt';actions=root/'actions';ready=root/'ready';trajectory=root/'trajectory'
        script=Path(__file__).with_name('ppo_policy_bridge.py')
        cmd=[sys.executable,'-u',str(script),'--state-file',str(states),'--action-dir',str(actions),
             '--trajectory-file',str(trajectory),'--frames','2','--seed','7','--ready-file',str(ready),
             '--idle-timeout',('.6' if kind=='truncated' else '5')]
        lines=['STATE '+str(p)+' '+' '.join(['.2']*44)+'\n' for p in range(2)]
        with (root/'log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            def wait_for(predicate,limit=35):
                end=time.monotonic()+limit
                while not predicate():
                    if proc.poll() is not None:raise AssertionError((root/'log').read_text())
                    if time.monotonic()>end:raise TimeoutError('bridge test deadline')
                    time.sleep(.005)
            try:
                wait_for(ready.exists)
                with states.open('w',encoding='utf-8') as stream:
                    if kind in ('malformed','truncated'):
                        stream.write(' '.join(lines[0].split()[:28])+('\n' if kind=='malformed' else ''));stream.flush()
                    else:
                        for p,line in enumerate(lines):
                            if kind=='split':
                                cut=line.index(' .2',line.index(' .2')+1)+3
                                stream.write(line[:cut]);stream.flush();time.sleep(.12)
                                self.assertIsNone(proc.poll());self.assertFalse((actions/f'{p}.txt').exists())
                                stream.write(line[cut:]);stream.flush()
                            else:stream.write(line);stream.flush()
                            wait_for(lambda:(actions/f'{p}.txt').exists())
                code=proc.wait(timeout=10);log.flush();return code,(root/'log').read_text(),trajectory.read_bytes()
            finally:
                if proc.poll() is None:proc.kill();proc.wait()

    def test_fragmented_lines_preserve_rng_and_trajectory(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=self.run_bridge(root/'complete','complete');b=self.run_bridge(root/'split','split')
            self.assertEqual(a[0],0,a[1]);self.assertEqual(b[0],0,b[1]);self.assertEqual(a[2],b[2])
            self.assertEqual(len(a[2].splitlines()),2)

    def test_malformed_complete_record_is_not_silently_skipped(self):
        with TemporaryDirectory() as folder:
            rc,log,_=self.run_bridge(folder,'malformed');self.assertNotEqual(rc,0);self.assertIn('got 28 fields',log)

    def test_unfinished_record_times_out_without_action(self):
        with TemporaryDirectory() as folder:
            rc,log,tr=self.run_bridge(folder,'truncated');self.assertNotEqual(rc,0);self.assertIn('TimeoutError',log);self.assertEqual(tr,b'')


if __name__=='__main__':main()
