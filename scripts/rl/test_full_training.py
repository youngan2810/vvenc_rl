"""Unit tests that do not require the Kaggle dataset or GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_train import (RATES, create_schedule, choose_window, atomic_checkpoint,
                        checkpoint_bundle, load_trainer, train_checkpoint,
                        atomic_json, TRAINING_SCHEMA)
from ppo_agent import PPOActorCritic, PPOTrainer, MODEL_VERSION, Transition


class FullTrainingTests(unittest.TestCase):
    def test_spatial_actor_keeps_ctu_locality(self):
        torch.manual_seed(0)
        model = PPOActorCritic().eval()
        global_state = torch.zeros(1, 8)
        spatial = torch.zeros(1, 3, 3, 4)
        altered = spatial.clone()
        altered[0, 0, 0, 0] = 10.0
        with torch.no_grad():
            original, _ = model(spatial, global_state)
            changed, _ = model(altered, global_state)
        delta = (changed.mean - original.mean).abs().flatten()
        self.assertEqual(tuple(delta.shape), (12,))
        self.assertGreater(delta.max().item(), 1e-5)
        self.assertGreater(delta.std().item(), 1e-6)

    def test_epoch_schedule_exactly_65_x_5(self):
        plan = create_schedule(seed=11, epochs=2)
        self.assertEqual(len(plan), 650)
        for epoch in (0, 1):
            items = [x for x in plan if x["epoch"] == epoch]
            pairs = {(x["train_index"], x["rate"]) for x in items}
            self.assertEqual(pairs, {(i, r) for i in range(65) for r in RATES})
        self.assertEqual(plan, create_schedule(seed=11, epochs=2))
        self.assertNotEqual(plan, create_schedule(seed=12, epochs=2))

    def test_window_deterministic_and_bounded(self):
        self.assertEqual(choose_window(7, 0, 1, 256, 256), 0)
        values = [choose_window(7, e, 1, 256, 1010) for e in range(8)]
        self.assertTrue(all(0 <= i <= 754 for i in values))
        self.assertGreater(len(set(values)), 1)
        with self.assertRaises(ValueError):
            choose_window(7, 0, 1, 256, 100)

    def test_checkpoint_schema_and_export(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints").mkdir()
            model = PPOActorCritic()
            trainer = PPOTrainer(model, epochs=5, minibatch_size=64)
            config = {"seed": 1, "epochs": 1, "model_version": MODEL_VERSION}
            ckpt = root / "checkpoints/ckpt_iter_000000.pt"
            atomic_checkpoint(ckpt, train_checkpoint(model, trainer, 0, config, "lock", {}))
            from real_reward_pilot import sha256_file
            progress = {"schema": TRAINING_SCHEMA, "latest_checkpoint":
                        str(ckpt.relative_to(root)), "latest_sha256": sha256_file(ckpt)}
            atomic_json(root / "progress.json", progress)
            self.assertTrue(checkpoint_bundle(root, progress).is_file())
            state, new_model, new_trainer = load_trainer(ckpt, config, "lock")
            self.assertEqual(state["ppo_iteration"], 0)
            self.assertEqual(state["model_version"], MODEL_VERSION)
            with self.assertRaises(ValueError):
                load_trainer(ckpt, {"different": True}, "lock")

    def test_resume_matches_uninterrupted_seeded_update(self):
        torch.manual_seed(17)
        model = PPOActorCritic()
        trainer = PPOTrainer(model, learning_rate=1e-4, epochs=2, minibatch_size=8)
        def collect():
            rows = []
            for i in range(16):
                sp, gl = torch.rand(1,3,3,4), torch.rand(1,8)
                with torch.no_grad():
                    raw, _, lp, val = model.sample(sp, gl)
                rows.append(Transition(sp[0], gl[0], raw[0], float(lp[0]),
                                       .2 + i/50, float(val[0]), 0., i == 15))
            return rows
        trainer.update(collect())
        with TemporaryDirectory() as folder:
            path = Path(folder)/'resume.pt'; cfg = {'source_commit': 'test-source'}
            atomic_checkpoint(path, train_checkpoint(model, trainer, 1, cfg, 'lock', {}))
            state, restored, other = load_trainer(path, cfg, 'lock')
            self.assertEqual(state['source_commit'], 'test-source')
            self.assertEqual(other.hyperparameters(), trainer.hyperparameters())
            rollout = collect()
            torch.manual_seed(123); first = trainer.update(rollout)
            torch.manual_seed(123); second = other.update(rollout)
            self.assertEqual(first, second)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()

