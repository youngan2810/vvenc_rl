import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, mock

import real_reward_pilot as pilot
import full_validate as val
import benchmark_cpu_threads as bench


class CodecThreadTests(TestCase):
    def test_benchmark_pairs_same_policy_windows_and_seeds(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder)
            encoder=root/'bin/release-static/vvencapp'
            encoder.parent.mkdir(parents=True); encoder.write_bytes(b'encoder')
            decoder=root/'decoder'; decoder.write_bytes(b'decoder')
            checkpoint=root/'policy.pt'; checkpoint.write_bytes(b'policy')
            args=SimpleNamespace(repo=root, decoder=decoder, checkpoint=checkpoint,
                output=root/'benchmark', train_indices=[0,1], frames=32,
                bitrate_kbps=120, repeats=2, seed=42, beta=.1)
            stack.enter_context(mock.patch.object(bench.torch.cuda,'is_available',return_value=True))
            stack.enter_context(mock.patch.object(bench.torch.cuda,'device_count',return_value=2))
            stack.enter_context(mock.patch.object(bench,'read_lock_manifest',return_value=({'train':['a','b']},{})))
            stack.enter_context(mock.patch.object(bench,'read_baseline_config',return_value={'lock_sha256':'locked'}))
            stack.enter_context(mock.patch.object(bench,'initialize_detector_assets'))
            stack.enter_context(mock.patch.object(bench,'dataset_source',return_value=root/'video.mov'))
            stack.enter_context(mock.patch.object(bench,'video_frame_capacity',return_value=300))
            stack.enter_context(mock.patch.object(bench.subprocess,'check_output',return_value='a'*40))
            calls=[]
            def collect(jobs, repo, decoder, checkpoint, seed, beta, **kwargs):
                calls.append((checkpoint, seed, [(j['video_id'],j['start_frame'],j['rate']) for j in jobs], kwargs))
                for job in jobs:
                    job['episode'].mkdir()
                    summary={'policy_checkpoint_sha256':bench.sha256_file(checkpoint),
                        'video_id':job['video_id'], 'start_frame':job['start_frame'],
                        'target_kbps':job['rate'], 'actual_kbps':120, 'mean_reward':.5,
                        'mean_f1_reward_term':.6, 'delta_qp_histogram':{'0':384},
                        'timings':{}, **kwargs}
                    (job['episode']/'summary.json').write_text(json.dumps(summary))
                return []
            stack.enter_context(mock.patch.object(bench,'spawn_parallel_collectors',side_effect=collect))
            bench.run(args)
            self.assertEqual([c[3]['encoder_threads'] for c in calls], [2,4,4,2])
            self.assertTrue(all(c[3]['encoder_threads']==c[3]['decoder_threads'] for c in calls))
            self.assertEqual(calls[0][:3], calls[1][:3])
            self.assertEqual(calls[2][:3], calls[3][:3])
            self.assertNotEqual(calls[0][1], calls[2][1])
            report=json.loads((args.output/'CPU_THREADS_BENCHMARK.json').read_text())
            self.assertTrue(report['no_optimizer_update'])
            self.assertEqual(len(report['runs']),4)

    def test_live_encode_passes_threads_and_accepts_all_zero_actions(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); out=root/'episode'; commands=[]; n=32
            class Bridge:
                def __init__(self,cmd,**kwargs):
                    Path(cmd[cmd.index('--ready-file')+1]).write_text('READY\n')
                def poll(self): return 0
                def wait(self,timeout=None): return 0
            def encode(cmd,log,**kwargs):
                commands.append(cmd); (out/'ppo.266').write_bytes(b'1234')
                rows=[{'poc':i,'discrete_action':[0]*12} for i in range(n)]
                (out/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with mock.patch.object(pilot.subprocess,'Popen',Bridge), mock.patch.object(
                    pilot,'run_cmd',side_effect=encode), mock.patch.object(pilot,'read_au_bits',
                    return_value=[SimpleNamespace(cumulative_bits=32)]*n):
                _, rows, _ = pilot.encode_with_ppo(root,root/'enc',root/'in',out,n,encoder_threads=2)
            self.assertEqual(len(rows),32)
            self.assertEqual(commands[0][commands[0].index('--threads')+1], '2')

    def test_vanilla_and_decoder_use_requested_threads(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); commands=[]
            with mock.patch.object(val,'run_cmd',side_effect=lambda cmd,*a,**k: commands.append(cmd)):
                stream=val.vanilla_encode(root/'enc',root/'input',root/'out',32,120,encoder_threads=2)
                stream.write_bytes(b'bits')
                with mock.patch.object(val,'training_objective_metrics',return_value={}), mock.patch('decoder_alignment.verify_decode_mapping'), mock.patch.object(val,'convert_10bit_to_8bit'), mock.patch.object(val,'detect_all'), mock.patch.object(
                        val,'get_counts',return_value={'f1':.5}):
                    val.evaluate_bitstream(root/'dec',None,stream,'video',root/'detections',32,root/'out',120,.1,decoder_threads=3)
            self.assertEqual(commands[0][commands[0].index('--threads')+1], '2')
            self.assertEqual(commands[1][commands[1].index('--threads')+1], '3')


if __name__ == '__main__': main()
