"""Fault-injection tests: recovery must not skip Val or reuse partial data."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, mock
import zipfile

from full_train import RATES, atomic_json, checkpoint_bundle, create_schedule
from training_session import (contained_path, ensure_epoch_validations, new_attempt,
                              restore_bundle, session_limit, validate_progress)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SessionTests(TestCase):
    def fixture(self, root):
        (root / 'checkpoints').mkdir()
        checkpoint = root / 'checkpoints/epoch.pt'
        checkpoint.write_bytes(b'epoch end weights')
        progress = {'training_config': {'session_version': 3}, 'locksha': 'lock',
                    'completed': [{'step': i+1, 'episode_dir': f'episodes/{i+1}'} for i in range(325)],
                    'validations': [], 'updates': [{'episode_steps': [325],
                      'checkpoint_path': 'checkpoints/epoch.pt', 'checkpoint_sha256': digest(checkpoint)}],
                    'latest_checkpoint': 'checkpoints/epoch.pt', 'latest_sha256': digest(checkpoint)}
        args = SimpleNamespace(run_validation=True, repo=root, decoder=root/'vvdec',
                               validation_frames=256, beta=.1)
        atomic_json(root/'progress.json', progress)
        return args, progress

    def result(self, progress):
        return {'schema': 'vvenc-ppo-full-validation/1', 'split': 'val', 'val_videos': 7,
                'targets_kbps': list(RATES), 'policy_sampling': 'deterministic',
                'no_optimizer_update': True, 'frames_per_sample': 256, 'beta': .1,
                'checkpoint_sha256': progress['latest_sha256'], 'baseline_lock_sha256': 'lock',
                'macro_ppo_score': .7, 'macro_vanilla_score': .6,
                'macro_ppo_discounted_return': 70., 'macro_vanilla_discounted_return': 60.,
                'macro_ppo_mean_training_reward': .65, 'macro_vanilla_mean_training_reward': .55,
                'gamma': .99, 'validation_version': 3,
                'checkpoint_selection_metric': 'macro_ppo_discounted_return',
                'encoder_threads': 4, 'decoder_threads': 4,
                'pairs': [{'val_index': i, 'target_kbps': r,
                    'ppo': {'discounted_return':70., 'mean_training_reward':.65},
                    'vanilla': {'discounted_return':60., 'mean_training_reward':.55}}
                    for i in range(7) for r in RATES]}

    def test_best_checkpoint_uses_discounted_training_return_not_clip_score(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,p=self.fixture(root)
            first=root/'validation_epoch_001';first.mkdir()
            atomic_json(first/'validation_summary.json',self.result(p))
            ensure_epoch_validations(args,root,p)
            second=root/'checkpoints/epoch2.pt';second.write_bytes(b'second epoch weights')
            p['completed'] += [{'step':i+1} for i in range(325,650)]
            p['updates'].append({'episode_steps':[650],'checkpoint_path':'checkpoints/epoch2.pt','checkpoint_sha256':digest(second)})
            p['latest_sha256']=digest(second);p['latest_checkpoint']='checkpoints/epoch2.pt'
            result=self.result(p)
            result['macro_ppo_score']=.2  # Clip score prefers epoch1.
            result['macro_ppo_discounted_return']=80.
            for row in result['pairs']:row['ppo']['discounted_return']=80.
            output=root/'validation_epoch_002';output.mkdir()
            atomic_json(output/'validation_summary.json',result)
            ensure_epoch_validations(args,root,p)
            self.assertEqual((root/'best_validation.pt').read_bytes(),second.read_bytes())
            self.assertEqual(p['validations'][-1]['selection_value'],80.)
            self.assertEqual(p['validations'][-1]['selection_metric'],'macro_ppo_discounted_return')

    def test_failed_validation_is_retried_before_next_epoch_without_deletion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args, p = self.fixture(root)
            attempts = []
            def fail(cmd, log):
                out = Path(cmd[cmd.index('--output')+1]); out.mkdir()
                (out/'partial.txt').write_text('keep me')
                attempts.append(out)
                raise RuntimeError('session interrupted')
            with self.assertRaises(RuntimeError):
                ensure_epoch_validations(args, root, p, fail)
            p = json.loads((root/'progress.json').read_text())
            def success(cmd, log):
                out = Path(cmd[cmd.index('--output')+1]); out.mkdir()
                attempts.append(out)
                atomic_json(out/'validation_summary.json', self.result(p))
            ensure_epoch_validations(args, root, p, success)
            self.assertEqual(len(p['validations']), 1)
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertEqual((attempts[0]/'partial.txt').read_text(), 'keep me')
            ensure_epoch_validations(args, root, p, mock.Mock(side_effect=AssertionError('reran Val')))
            self.assertEqual((root/'best_validation.pt').read_bytes(), b'epoch end weights')

    def test_completed_summary_is_recovered_without_rerunning_encoder(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); args, p = self.fixture(root)
            out = root/'validation_epoch_001'; out.mkdir()
            atomic_json(out/'validation_summary.json', self.result(p))
            ensure_epoch_validations(args, root, p, mock.Mock(side_effect=AssertionError('should reuse summary')))
            self.assertEqual(p['validations'][0]['epoch'], 1)

    def test_wrong_checkpoint_validation_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); args, p = self.fixture(root)
            out = root/'validation_epoch_001'; out.mkdir()
            result = self.result(p); result['checkpoint_sha256'] = 'wrong'
            atomic_json(out/'validation_summary.json', result)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                ensure_epoch_validations(args, root, p)
            self.assertEqual(p['validations'], [])

    def test_pending_epoch_weights_and_summary_survive_archive_restore(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'run'; root.mkdir(); args, p = self.fixture(root)
            out = root/'validation_epoch_001'; out.mkdir()
            atomic_json(out/'validation_summary.json', self.result(p))
            p['pending_validation'] = {'epoch': 1, 'output': out.name}
            atomic_json(root/'progress.json', p)
            archive = checkpoint_bundle(root, p)
            restored = Path(folder)/'restored'
            restore_bundle(archive, restored)
            q = json.loads((restored/'progress.json').read_text())
            ensure_epoch_validations(args, restored, q, mock.Mock(side_effect=AssertionError('reran')))
            self.assertEqual(len(q['validations']), 1)
            with self.assertRaises(ValueError):
                restore_bundle(archive, restored)

    def test_attempt_preserves_existing_checkpoint(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'ckpt.pt'; path.write_bytes(b'old')
            retry = new_attempt(path)
            self.assertNotEqual(retry, path); self.assertFalse(retry.exists())
            self.assertEqual(path.read_bytes(), b'old')

    def test_session_limit_preserves_pairing_at_epoch_boundary(self):
        self.assertEqual(session_limit(324, 975, 2), 325)
        self.assertEqual(session_limit(325, 975, 2), 327)
        self.assertEqual(session_limit(300, 975, 100), 399)
        self.assertEqual(session_limit(974, 975, 2), 975)
        self.assertEqual(session_limit(22, 975, 0), 975)

    def test_archive_path_traversal_rejected_without_writes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for bad in ('../evil', '/absolute', 'C:/absolute', 'a\\..\\evil'):
                with self.assertRaises(ValueError):
                    contained_path(root, bad)
            archive = root/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z: z.writestr('../evil', 'bad')
            with self.assertRaises(ValueError): restore_bundle(archive, root/'out')
            self.assertFalse((root/'out').exists())

    def test_progress_validates_schedule_and_chain(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); checkpoint = root/'latest.pt'; checkpoint.write_bytes(b'ok')
            cfg = {'seed': 7, 'epochs': 1}
            plan = create_schedule(7, 1)
            p = {'training_config': cfg, 'locksha': 'lock', 'completed': [], 'validations': [],
                 'latest_checkpoint': checkpoint.name, 'latest_sha256': digest(checkpoint),
                 'updates': [{'ppo_iteration': 1, 'episode_steps': [1,2],
                     'collector_checkpoint_sha256': 'initial', 'checkpoint_sha256': digest(checkpoint)}]}
            for i, item in enumerate(plan[:2]):
                p['completed'].append({'step': i+1, 'epoch': item['epoch']+1,
                      'train_index': item['train_index'], 'target_kbps': item['rate']})
            validate_progress(root, p, cfg, 'lock')
            p['completed'][1]['step'] = 1
            with self.assertRaises(ValueError): validate_progress(root, p, cfg, 'lock')


if __name__ == '__main__': main()

