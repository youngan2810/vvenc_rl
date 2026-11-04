"""Deterministic sanity tests for PPO math and Stage-2 tensor interfaces.

Run: python3 -m unittest discover -s scripts/rl -p 'test_ppo_*.py'
"""
import math
import unittest

import torch

from ppo_agent import PPOActorCritic, PPOTrainer, Transition, compute_gae, rate_f1_reward


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PPOActorCritic()
        self.spatial = torch.rand(4, 3, 3, 4)
        self.global_ = torch.rand(4, 8)

    def test_actor_critic_shapes_actions_and_logprob(self):
        raw, action, logp, value = self.model.sample(self.spatial, self.global_)
        self.assertEqual(tuple(raw.shape), (4, 12))
        self.assertEqual(tuple(action.shape), (4, 12))
        self.assertEqual(tuple(value.shape), (4,))
        self.assertTrue(torch.all(action >= -2))
        self.assertTrue(torch.all(action <= 2))
        self.assertTrue(torch.all(action == action.round()))
        self.assertTrue(torch.isfinite(logp).all())
        eval_logp, entropy, eval_value = self.model.evaluate(self.spatial, self.global_, raw)
        self.assertTrue(torch.allclose(logp, eval_logp))
        self.assertTrue(torch.allclose(value, eval_value))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_gae_episode_boundary(self):
        reward = torch.tensor([1.0, 1.0, 10.0])
        zeros = torch.zeros(3)
        done = torch.tensor([0.0, 1.0, 1.0])
        adv, ret = compute_gae(reward, zeros, zeros, done, gamma=0.9, gae_lambda=1.0)
        # The third episode reward must not leak into the first two steps.
        self.assertTrue(torch.allclose(adv, torch.tensor([1.9, 1.0, 10.0])))
        self.assertTrue(torch.allclose(ret, adv))

    def test_ppo_training_updates_weights(self):
        rollout = []
        for i in range(16):
            sp = torch.rand(1, 3, 3, 4)
            gl = torch.rand(1, 8)
            with torch.no_grad():
                raw, action, lp, val = self.model.sample(sp, gl)
            rollout.append(Transition(
                spatial=sp.squeeze(0), global_=gl.squeeze(0),
                raw_action=raw.squeeze(0), log_prob=float(lp.item()),
                reward=float((i % 4) / 4.0),
                value=float(val.item()), next_value=0.0,
                done=(i % 4 == 3),
            ))
        trainer = PPOTrainer(self.model, epochs=2, minibatch_size=8)
        before = self.model.actor_cells[-1].weight.detach().clone()
        output = trainer.update(rollout)
        self.assertTrue(math.isfinite(output["loss"]))
        self.assertGreater(output["optimizer_steps"], 0)
        self.assertFalse(torch.equal(before, self.model.actor_cells[-1].weight.detach()))

    def test_rate_reward_constraints(self):
        self.assertAlmostEqual(
            rate_f1_reward(0.8, 120000, 120000, 1.0, 0.5), 0.8
        )
        self.assertLess(
            rate_f1_reward(0.8, 240000, 120000, 1.0, 0.5), 0.8
        )
        with self.assertRaises(ValueError):
            rate_f1_reward(1.5, 100, 120000, 1.0, 0.5)

    def test_kl_stop_does_not_apply_another_optimizer_step(self):
        sp, gl = self.spatial[:1], self.global_[:1]
        with torch.no_grad():
            raw, _, lp, val = self.model.sample(sp, gl)
        # Simulate a policy already far from its collector. No gradient is
        # allowed on this minibatch, unlike the previous post-step check.
        row = Transition(sp[0], gl[0], raw[0], float(lp[0])+1.0,
                         .5, float(val[0]), 0., True)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        report = PPOTrainer(self.model).update([row])
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertEqual(report['early_stop'], 1)
        self.assertGreater(report['stop_kl'], .03)
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k, v in before.items()))


if __name__ == "__main__":
    unittest.main()

