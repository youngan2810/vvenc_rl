"""Deterministic sanity tests for PPO math and Stage-2 tensor interfaces.

Run: python3 -m unittest discover -s scripts/rl -p 'test_ppo_*.py'
"""
import math
import unittest

import torch

from ppo_agent import PPOActorCritic, PPOTrainer, Transition, compute_gae, rate_f1_reward, estimate_kl


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

    def test_kl_estimators_nonnegative_k3_and_signed_k1(self):
        ratios = torch.tensor([-.2, 0., .2], dtype=torch.float64, requires_grad=True)
        k3, k1 = estimate_kl(ratios)
        self.assertGreater(k3.item(), 0)
        self.assertAlmostEqual(k1.item(), 0)
        self.assertFalse(k3.requires_grad)
        self.assertFalse(k1.requires_grad)
        zero, _ = estimate_kl(torch.zeros(3))
        self.assertEqual(zero.item(), 0)
        tiny, _ = estimate_kl(torch.tensor([-1e-8, 1e-8], dtype=torch.float64))
        self.assertGreaterEqual(tiny.item(), 0)

    def test_kl_k3_matches_gaussian_analytic_kl_in_expectation(self):
        # Deterministic quantile quadrature under old N(0,1), new N(.2,1).
        # Exercise the production estimator, not a duplicate of its formula.
        u = (torch.arange(100000, dtype=torch.float64) + .5) / 100000
        z = math.sqrt(2) * torch.erfinv(2*u-1)
        log_ratio = .2*z - .02
        k3, k1 = estimate_kl(log_ratio)
        self.assertAlmostEqual(k3.item(), .02, delta=1e-5)
        self.assertAlmostEqual(k1.item(), .02, places=7)

    def test_positive_log_ratio_can_stop_even_when_signed_kl_negative(self):
        sp, gl = self.spatial[:1], self.global_[:1]
        with torch.no_grad():
            raw, _, lp, val = self.model.sample(sp, gl)
        row = Transition(sp[0], gl[0], raw[0], float(lp[0])-1., .5, float(val[0]), 0., True)
        report = PPOTrainer(self.model).update([row])
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertGreater(report['stop_kl'], .03)
        self.assertLess(report['stop_kl_signed'], 0)


if __name__ == "__main__":
    unittest.main()


