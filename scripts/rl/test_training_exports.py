import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
import zipfile

from export_training_results import export_analysis
from full_train import atomic_json, checkpoint_bundle
from real_reward_pilot import sha256_file


class ExportTests(TestCase):
    def test_resume_zip_excludes_logs_and_rollouts_analysis_keeps_them(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); (root/'checkpoints').mkdir()
            checkpoint = root/'checkpoints/latest.pt'; checkpoint.write_bytes(b'weights')
            ep = root/'episodes/one'; ep.mkdir(parents=True)
            (ep/'joined_reward_ppo.jsonl').write_text('{"reward":1}\n'*10000)
            (ep/'source.yuv').write_bytes(b'raw image')
            (ep/'encoder.log').write_text('encoder evidence')
            atomic_json(ep/'summary.json', {'reward':1})
            validation = root/'validation_epoch_001'; validation.mkdir()
            atomic_json(validation/'validation_progress.json', {'completed':[{'target':50}]})
            progress = {'latest_checkpoint':'checkpoints/latest.pt', 'latest_sha256':sha256_file(checkpoint),
                        'completed':[{'step':1, 'episode_dir':'episodes/one'}], 'updates':[]}
            atomic_json(root/'progress.json', progress)
            with zipfile.ZipFile(checkpoint_bundle(root, progress)) as z:
                self.assertIn('checkpoint/latest.pt', z.namelist())
                self.assertIn('validation_epoch_001/validation_progress.json', z.namelist())
                self.assertFalse(any(name.startswith('episodes/') for name in z.namelist()))
            with zipfile.ZipFile(export_analysis(root)) as z:
                self.assertIn('episodes/one/joined_reward_ppo.jsonl', z.namelist())
                self.assertIn('episodes/one/encoder.log', z.namelist())
                self.assertNotIn('episodes/one/source.yuv', z.namelist())
                self.assertFalse(any(name.endswith('.pt') for name in z.namelist()))


if __name__ == '__main__': main()
