import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

import torch
import bridge_io_upgrade as upgrade
from full_train import (TRAINING_SCHEMA, atomic_checkpoint, atomic_json, checkpoint_bundle,
                        create_schedule, train_checkpoint)
from full_train_dual_gpu import batch_sizes
from ppo_agent import PPOActorCritic, PPOTrainer
from training_session import restore_bundle, validate_progress


class UpgradeTests(TestCase):
    def fixture(self, root):
        torch.set_num_threads(2)
        cfg=dict(seed=7, epochs=3, source_commit=upgrade.BASE,
                 collector_runner_sha256='old-runner', session_version=3, beta=.1)
        new=dict(cfg, source_commit='f'*40, collector_runner_sha256='new-runner')
        path=root/'checkpoints/latest.pt';path.parent.mkdir()
        model=PPOActorCritic();trainer=PPOTrainer(model)
        weights=train_checkpoint(model,trainer,236,cfg,upgrade.LOCK,{})
        atomic_checkpoint(path,weights)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        plan=create_schedule(7,3)
        progress=dict(schema=TRAINING_SCHEMA,training_config=cfg,locksha=upgrade.LOCK,
            completed=[dict(step=i+1,epoch=x['epoch']+1,train_index=x['train_index'],target_kbps=x['rate']) for i,x in enumerate(plan[:471])],
            validations=[],updates=[dict(ppo_iteration=i+1,episode_steps=list(range(start+1,start+size+1)),
                collector_checkpoint_sha256=digest,checkpoint_sha256=digest,checkpoint_path='checkpoints/latest.pt')
                for i,(start,size) in enumerate(batch_sizes(0,471))],
            latest_checkpoint='checkpoints/latest.pt',latest_sha256=digest)
        atomic_json(root/'progress.json',progress)
        return cfg,new,progress,path,digest

    def migrate(self,root,progress,new,digest):
        with mock.patch.object(upgrade,'CHECKPOINT',digest),mock.patch.object(upgrade,'check_patch',return_value=(new['source_commit'],'diff')):
            return upgrade.upgrade_progress(root,root,progress,new,upgrade.LOCK)

    def test_preserves_checkpoint_history_and_portable_preupdate_resume(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old,new,p,path,digest=self.fixture(root);before=path.read_bytes()
            migrated=self.migrate(root,p,new,digest)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(migrated['completed'],p['completed']);self.assertEqual(migrated['updates'],p['updates'])
            self.assertEqual(migrated['bridge_io_upgrade']['from_config'],old)
            self.assertEqual(json.loads((root/'progress_before_bridge_io_upgrade.json').read_text()),p)
            validate_progress(root,migrated,new,upgrade.LOCK)
            restored=root/'restored';restore_bundle(root/'PPO_QP_FULL_TRAIN_LATEST.zip',restored)
            with mock.patch.object(upgrade,'CHECKPOINT',digest):
                saved,model,trainer=upgrade.load_compatible_trainer(restored/migrated['latest_checkpoint'],new,upgrade.LOCK,migrated)
            self.assertEqual(saved['training_config'],old);self.assertEqual(saved['ppo_iteration'],236)
            self.assertEqual((restored/migrated['latest_checkpoint']).read_bytes(),before)
            self.assertEqual(trainer.optimizer.state_dict()['param_groups'],saved['optimizer_state']['param_groups'])
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,saved['model_state'][key]))
            self.assertEqual(upgrade.upgrade_progress(root,root,migrated,new,upgrade.LOCK),migrated)

    def test_new_checkpoint_loads_new_config_after_first_update(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old,new,p,path,digest=self.fixture(root)
            migrated=self.migrate(root,p,new,digest)
            model=PPOActorCritic();trainer=PPOTrainer(model)
            other=root/'checkpoints/new.pt';atomic_checkpoint(other,train_checkpoint(model,trainer,237,new,upgrade.LOCK,{}))
            with mock.patch.object(upgrade,'CHECKPOINT',digest):
                obj,_,_=upgrade.load_compatible_trainer(other,new,upgrade.LOCK,migrated)
            self.assertEqual(obj['training_config'],new)

    def test_changed_reward_or_seed_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old,new,p,path,digest=self.fixture(root)
            for key,value in [('beta',.2),('seed',8),('epochs',2)]:
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'unchanged PPO settings'):
                    self.migrate(root,p,dict(new,**{key:value}),digest)
            self.assertFalse((root/'progress_before_bridge_io_upgrade.json').exists())

    def test_wrong_checkpoint_or_episode_count_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old,new,p,path,digest=self.fixture(root)
            bad=copy.deepcopy(p);bad['completed'].pop()
            with self.assertRaisesRegex(ValueError,'EP471'):self.migrate(root,bad,new,digest)
            bad=copy.deepcopy(p);bad['latest_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'EP471'):self.migrate(root,bad,new,digest)

    def test_audit_tampering_rejected_by_loader(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old,new,p,path,digest=self.fixture(root)
            migrated=self.migrate(root,p,new,digest)
            migrated['bridge_io_upgrade']['from_config']['beta']=.2
            with mock.patch.object(upgrade,'CHECKPOINT',digest),self.assertRaisesRegex(ValueError,'provenance'):
                upgrade.load_compatible_trainer(path,new,upgrade.LOCK,migrated)

    def test_source_scope_rejects_algorithm_change(self):
        def fake(repo,*args):
            if args==('rev-parse','HEAD'):return b'f'*40
            if args[0]=='status':return b''
            if args[:2]==('diff','--name-only'):return b'scripts/rl/ppo_agent.py\n'
            raise AssertionError(args)
        with mock.patch.object(upgrade,'git',side_effect=fake),self.assertRaisesRegex(ValueError,'outside the IO repair'):
            upgrade.check_patch(Path('unused'))


if __name__=='__main__':main()
