import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import alignment_upgrade as u


class UpgradeTests(unittest.TestCase):
    def fixture(self):
        original={'source_commit':u.BASE,'collector_runner_sha256':'old','frames':256}
        cfg=dict(original, source_commit='a'*40,collector_runner_sha256='new')
        p=dict(training_config=original,locksha=u.LOCK,completed=[{}]*58,
               updates=[{}]*29,latest_sha256=u.CHECKPOINT,latest_checkpoint='checkpoint.pt')
        return p,cfg

    def test_upgrade_preserves_original_and_checkpoint_and_is_idempotent(self):
        p,cfg=self.fixture();before=copy.deepcopy(p)
        with TemporaryDirectory() as folder, mock.patch('training_session.validate_progress') as validate, \
                mock.patch('full_train.load_trainer') as load, mock.patch('full_train.checkpoint_bundle'), \
                mock.patch.object(u,'check_patch',return_value=(cfg['source_commit'],'diffhash')):
            root=Path(folder);checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'unchanged checkpoint')
            h=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result=u.upgrade_progress(root,root,p,cfg,u.LOCK)
            validate.assert_called_once_with(root,p,p['training_config'],u.LOCK)
            load.assert_called_once_with(checkpoint,p['training_config'],u.LOCK)
            self.assertEqual(p,before)
            self.assertEqual(json.loads((root/'progress_before_alignment_upgrade.json').read_text()),before)
            self.assertEqual(result['alignment_upgrade']['from_config'],before['training_config'])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),h)
            self.assertEqual(u.upgrade_progress(root,root,result,cfg,u.LOCK),result)

    def test_changed_settings_or_checkpoint_or_boundary_are_rejected(self):
        for field,value in [('latest_sha256','bad'),('completed',[]),('updates',[])]:
            p,cfg=self.fixture();p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                u.upgrade_progress(Path('.'),Path('.'),p,cfg,u.LOCK)
        p,cfg=self.fixture();cfg['frames']=128
        with self.assertRaises(ValueError):u.upgrade_progress(Path('.'),Path('.'),p,cfg,u.LOCK)

    def test_checkpoint_loader_uses_old_config_only_for_exact_original_hash(self):
        p,cfg=self.fixture()
        p['alignment_upgrade']=dict(schema='vvenc-alignment-upgrade/1',from_config=p['training_config'],
                                    to_config=cfg,checkpoint_sha256=u.CHECKPOINT)
        with mock.patch('real_reward_pilot.sha256_file',return_value=u.CHECKPOINT), \
                mock.patch('full_train.load_trainer',return_value='loaded') as load:
            self.assertEqual(u.load_compatible_trainer(Path('c'),cfg,u.LOCK,p),'loaded')
            load.assert_called_with(Path('c'),p['training_config'],u.LOCK)
            p['alignment_upgrade']['to_config']={}
            with self.assertRaises(ValueError):u.load_compatible_trainer(Path('c'),cfg,u.LOCK,p)
        with mock.patch('real_reward_pilot.sha256_file',return_value='newcheckpoint'), \
                mock.patch('full_train.load_trainer') as load:
            u.load_compatible_trainer(Path('c'),cfg,u.LOCK,p)
            load.assert_called_once_with(Path('c'),cfg,u.LOCK)

    def test_algorithm_file_change_is_rejected(self):
        def git(repo,*args):
            if args[0]=='rev-parse':return b'a'*40
            if args[0]=='status':return b''
            if args[:2]==('diff','--name-only'):return b'scripts/rl/ppo_agent.py\n'
            raise AssertionError(args)
        with mock.patch.object(u,'git',side_effect=git),self.assertRaisesRegex(ValueError,'unapproved'):
            u.check_patch(Path('.'))


if __name__=='__main__':unittest.main()
