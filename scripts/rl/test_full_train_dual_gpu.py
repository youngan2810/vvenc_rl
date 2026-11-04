"""Two-GPU episode concurrency and PPO on-policy invariants (CPU-only)."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_train_dual_gpu import (COLLECTORS, EPISODES_PER_EPOCH,
                                batch_sizes, spawn_parallel_collectors)
from full_train import create_schedule
from ppo_agent import PPOActorCritic, PPOTrainer, Transition, compute_gae


class DualGpuTests(TestCase):
    def test_batches_do_not_cross_epochs(self):
        sizes = list(batch_sizes(0, 975))
        self.assertEqual(sum(n for _, n in sizes), 975)
        self.assertEqual(len(sizes), 3 * 163)  # 162 pairs + 1 single per epoch
        self.assertEqual(sizes[0], (0, 2))
        self.assertEqual(sizes[162], (324, 1))
        self.assertEqual(sizes[163], (325, 2))
        self.assertEqual(sizes[-1], (974, 1))
        for start, size in sizes:
            self.assertLessEqual(size, COLLECTORS)
            self.assertEqual(start // EPISODES_PER_EPOCH,
                             (start + size - 1) // EPISODES_PER_EPOCH)

    def test_training_schedule_has_five_bitrates_each_video(self):
        p = create_schedule(20261103, 3)
        self.assertEqual(len(p), 975)
        self.assertEqual(len({(x['train_index'], x['rate']) for x in p[:325]}), 325)

    def test_two_gpu_process_environment_and_same_checkpoint(self):
        # Mock only subprocess spawning; no actual GPU required for this test.
        with TemporaryDirectory() as folder:
            root = Path(folder)
            ckpt = root / 'same_frozen_policy.pt'
            ckpt.write_bytes(b'checkpoint')
            jobs = [dict(step=i+1, train_index=i, rate=50+i*30,
                         frames=64, start_frame=5, episode=root/f'episode{i}',
                         log=root/f'collector{i}.log') for i in range(2)]
            captured = []
            class DummyProcess:
                pid = 100
                def poll(self): return 0
                def terminate(self): pass
                def wait(self, timeout=None): return 0
                def kill(self): pass
            def fake_popen(cmd, *, stdout, stderr, env, start_new_session):
                captured.append((cmd, dict(env)))
                return DummyProcess()
            with mock.patch('full_train_dual_gpu.subprocess.Popen', side_effect=fake_popen):
                results = spawn_parallel_collectors(jobs, root, ckpt, ckpt,
                                                     seed=7, beta=.1)
            self.assertEqual(len(results), 2)
            self.assertEqual([env['CUDA_VISIBLE_DEVICES'] for _, env in captured],
                             ['0', '1'])
            self.assertTrue(all('--policy-checkpoint' in cmd for cmd, env in captured))
            self.assertEqual([cmd[cmd.index('--policy-checkpoint')+1] for cmd, _ in captured],
                             [str(ckpt)]*2)

    def test_gae_cuts_across_two_videos(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.zeros(4)
        next_values = torch.zeros(4)
        terminal = torch.tensor([0., 1., 0., 1.])
        joined, _ = compute_gae(rewards, values, next_values, terminal,
                                gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.equal(joined, torch.tensor([3.0,2.0,7.0,4.0])))

    def test_failed_worker_cleans_up_linux_process_groups(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = [dict(step=i+1, train_index=i, rate=50, frames=64,
                         start_frame=0, episode=root/f'episode{i}', log=root/f'{i}.log') for i in range(2)]
            processes = []
            def spawn(*args, **kwargs):
                proc = mock.Mock(pid=100+len(processes))
                proc.poll.return_value = 1 if not processes else None
                processes.append(proc)
                return proc
            fake_os = SimpleNamespace(name='posix', environ={}, killpg=mock.Mock())
            with mock.patch('full_train_dual_gpu.os', fake_os), mock.patch(
                    'full_train_dual_gpu.signal', SimpleNamespace(SIGTERM=15, SIGKILL=9)), mock.patch(
                    'full_train_dual_gpu.subprocess.Popen', side_effect=spawn):
                with self.assertRaises(RuntimeError):
                    spawn_parallel_collectors(jobs, root, root/'decoder', root/'policy', 7, .1)
            self.assertEqual({call.args[0] for call in fake_os.killpg.call_args_list}, {100, 101})

    def test_combined_rollout_is_one_ppo_update(self):
        torch.manual_seed(3)
        model = PPOActorCritic()
        trainer = PPOTrainer(model, epochs=1, minibatch_size=16)
        transitions = []
        for episode in range(2):
            for i in range(4):
                spatial = torch.randn(3, 3, 4)
                global_ = torch.rand(8)
                with torch.no_grad():
                    raw, action, log_prob, value = model.sample(
                        spatial.unsqueeze(0), global_.unsqueeze(0))
                transitions.append(Transition(spatial, global_, raw[0],
                                              float(log_prob[0]), reward=0.2+i/10,
                                              value=float(value[0]), next_value=0.,
                                              done=(i == 3)))
        report = trainer.update(transitions)
        self.assertEqual(report['optimizer_steps'], 1.0)


if __name__ == '__main__':
    unittest.main()
