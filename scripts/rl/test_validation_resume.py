"""Exercise the actual validation loop with mocked codecs/detector and failures."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, mock
from contextlib import ExitStack

import full_validate as val
from full_train import TRAINING_SCHEMA, atomic_json, checkpoint_bundle
from ppo_agent import MODEL_VERSION
from training_session import restore_bundle


class ValidationResumeTests(TestCase):
    def test_pair_resume_survives_portable_archive_without_raw_files(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)/'run'; (root/'checkpoints').mkdir(parents=True)
            checkpoint = root/'checkpoints/epoch.pt'; checkpoint.write_bytes(b'weights')
            encoder = root/'bin/release-static/vvencapp'
            encoder.parent.mkdir(parents=True); encoder.write_bytes(b'encoder')
            (root/'decoder').write_bytes(b'decoder')
            args = SimpleNamespace(repo=root, decoder=root/'decoder', checkpoint=checkpoint,
                output=root/'validation_epoch_001', frames=32, val_videos=1, beta=.1,
                seed=42, encoder_threads=2, decoder_threads=2, resume_bundle_root=root)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            progress = {'training_config':{'session_version':3}, 'completed':[],
                'latest_checkpoint':'checkpoints/epoch.pt', 'latest_sha256':digest,
                'updates':[{'episode_steps':[325], 'checkpoint_path':'checkpoints/epoch.pt',
                            'checkpoint_sha256':digest}],
                'pending_validation':{'epoch':1, 'output':'validation_epoch_001'}}
            atomic_json(root/'progress.json', progress)
            weights = {'schema': TRAINING_SCHEMA, 'model_version': MODEL_VERSION,
                       'baseline_lock_sha256': 'lock',
                       'training_config': {'encoder_threads': 2, 'decoder_threads': 2}}
            stack.enter_context(mock.patch.object(val.torch.cuda, 'is_available', return_value=True))
            stack.enter_context(mock.patch.object(val.torch, 'load', return_value=weights))
            stack.enter_context(mock.patch.object(val, 'read_lock_manifest',
                return_value=({'train':['train'], 'val':['held_out'], 'test':['test']}, {})))
            stack.enter_context(mock.patch.object(val, 'read_baseline_config', return_value={'lock_sha256':'lock'}))
            detector_loader = stack.enter_context(mock.patch.object(val, 'load_frozen_yolov5',
                                                     return_value=(object(), {'commit':'frozen'})))
            stack.enter_context(mock.patch.object(val, 'dataset_source', return_value=Path('video.mov')))
            def preprocess(source, dest, frames, log, **kwargs): dest.write_bytes(b'yuv')
            def detect(model, source, frames, vid, dest): dest.write_text('{}\n')
            stack.enter_context(mock.patch.object(val, 'preprocess_mov', side_effect=preprocess))
            stack.enter_context(mock.patch.object(val, 'detect_all', side_effect=detect))
            vanilla_rates, ppo_rates, threads = [], [], []
            def vanilla(encoder, source, output, frames, rate, encoder_threads):
                output.mkdir(); stream=output/'vanilla.266'; stream.write_bytes(b'bits')
                vanilla_rates.append(rate); threads.append(encoder_threads)
                return stream
            def ppo(repo, encoder, source, output, frames, **kwargs):
                output.mkdir(); stream=output/'ppo.266'; stream.write_bytes(b'bits')
                ppo_rates.append(kwargs['bitrate_kbps']); threads.append(kwargs['encoder_threads'])
                return stream, [None]*frames, [None]*frames
            fail = [True]
            def evaluate(decoder, detector, stream, vid, source_det, frames, out, rate, beta, decoder_threads):
                threads.append(decoder_threads)
                if stream.name == 'ppo.266' and rate == 120 and fail[0]:
                    raise RuntimeError('simulated interruption')
                f1 = .7 if stream.name == 'ppo.266' else .6
                return {'tp':7, 'fp':2, 'fn':1, 'precision':.7, 'recall':.7,
                        'f1':f1, 'actual_kbps':rate, 'target_kbps':rate, 'rate_penalty':0., 'score':f1}
            stack.enter_context(mock.patch.object(val, 'vanilla_encode', side_effect=vanilla))
            stack.enter_context(mock.patch.object(val, 'encode_with_ppo', side_effect=ppo))
            stack.enter_context(mock.patch.object(val, 'evaluate_bitstream', side_effect=evaluate))
            with self.assertRaisesRegex(RuntimeError, 'interruption'): val.run(args)
            state = json.loads((args.output/'validation_progress.json').read_text())
            self.assertEqual([x['target_kbps'] for x in state['completed']], [50,80])
            self.assertTrue((args.output/'val_00/rate_120/ppo/ppo.266').is_file())

            # The ZIP already contains the completed pairs: no finalizer/export
            # is needed after a simulated hard interruption.
            archive = root/'PPO_QP_FULL_TRAIN_LATEST.zip'
            fresh = Path(folder)/'restored'; restore_bundle(archive, fresh)
            args.resume_bundle_root = fresh
            args.output = fresh/'validation_epoch_001'; args.checkpoint = fresh/'checkpoints/epoch.pt'
            self.assertFalse((args.output/'val_00').exists())
            fail[0] = False; vanilla_rates.clear(); ppo_rates.clear()
            val.run(args)
            self.assertEqual(vanilla_rates, [120,160,200])
            self.assertEqual(ppo_rates, [120,160,200])
            self.assertEqual(set(threads), {2})
            result = json.loads((args.output/'validation_summary.json').read_text())
            self.assertEqual(len(result['pairs']), 5)
            self.assertAlmostEqual(result['macro_delta_f1'], .1)
            # A completed run can regenerate its summary without any new encode.
            detector_loader.reset_mock(); vanilla_rates.clear(); val.run(args)
            detector_loader.assert_not_called(); self.assertEqual(vanilla_rates, [])
            args.beta = .2
            with self.assertRaisesRegex(ValueError, 'changed checkpoint'): val.run(args)


if __name__ == '__main__': main()
