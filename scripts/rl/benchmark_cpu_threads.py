#!/usr/bin/env python3
"""Compare two concurrent collectors at 2/2 vs 4/4 codec threads; no PPO update."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from full_train import atomic_json, choose_window, clean_episode, video_frame_capacity
from full_train_dual_gpu import initialize_detector_assets, spawn_parallel_collectors
from real_reward_pilot import dataset_source, read_baseline_config, read_lock_manifest, sha256_file


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Benchmark requires two visible GPUs')
    if args.output.exists():
        raise FileExistsError('Use a new benchmark output directory')
    if args.repeats < 1 or not 32 <= args.frames <= 512 or any(i not in range(65) for i in args.train_indices):
        raise ValueError('Invalid benchmark repeats/frames/Train indices')
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    encoder = args.repo / 'bin/release-static/vvencapp'
    for path in (args.checkpoint, encoder, args.decoder):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    splits, records = read_lock_manifest(Path('/kaggle/input'))
    lock = read_baseline_config(Path('/kaggle/input'))
    initialize_detector_assets(args.repo, lock)
    jobs_template = []
    for i, idx in enumerate(args.train_indices):
        vid = splits['train'][idx]
        cap = video_frame_capacity(dataset_source(records, vid))
        jobs_template.append({'step': i+1, 'train_index': idx, 'video_id': vid,
                              'rate': args.bitrate_kbps, 'frames': args.frames,
                              'start_frame': choose_window(args.seed, 0, idx, args.frames, cap)})
    args.output.mkdir(parents=True)
    result = {'schema': 'vvenc-thread-benchmark/1', 'no_optimizer_update': True,
              'checkpoint_sha256': sha256_file(args.checkpoint),
              'encoder_sha256': sha256_file(encoder), 'decoder_sha256': sha256_file(args.decoder),
              'baseline_lock_sha256': lock['lock_sha256'], 'cpu_count': os.cpu_count(),
              'cpu_affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'source_commit': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'frames_per_episode': args.frames, 'seed': args.seed, 'beta': args.beta,
              'runs': []}
    for repeat in range(args.repeats):
        # Alternate order to reduce a systematic first-run/cache advantage.
        for threads in ((2, 4) if repeat % 2 == 0 else (4, 2)):
            group = args.output / f'repeat_{repeat+1:02d}_threads_{threads}'
            group.mkdir()
            jobs = [{**row, 'episode': group / f'worker_{i}', 'log': group / f'worker_{i}.log'}
                    for i, row in enumerate(jobs_template)]
            began = time.monotonic()
            spawn_parallel_collectors(jobs, args.repo, args.decoder, args.checkpoint,
                                      args.seed + repeat*1000, args.beta,
                                      encoder_threads=threads, decoder_threads=threads)
            elapsed = time.monotonic() - began
            summaries = [json.loads((job['episode'] / 'summary.json').read_text()) for job in jobs]
            for job, summary in zip(jobs, summaries):
                if (summary['policy_checkpoint_sha256'] != result['checkpoint_sha256']
                        or summary['video_id'] != job['video_id']
                        or summary['encoder_threads'] != threads or summary['decoder_threads'] != threads):
                    raise ValueError('Benchmark collector provenance mismatch')
            result['runs'].append({'repeat': repeat+1, 'encoder_threads': threads,
                                   'decoder_threads': threads, 'wall_seconds': elapsed,
                                   'episodes': [{k: s[k] for k in (
                                       'video_id', 'start_frame', 'target_kbps', 'actual_kbps',
                                       'mean_reward', 'mean_f1_reward_term', 'delta_qp_histogram', 'timings')}
                                       for s in summaries]})
            atomic_json(args.output / 'CPU_THREADS_BENCHMARK.json', result)
            for job in jobs:
                clean_episode(job['episode'], keep_yuv=False)
                (job['episode'] / 'online/ppo.266').unlink(missing_ok=True)
    result['median_pair_seconds'] = {str(t): statistics.median(
        row['wall_seconds'] for row in result['runs'] if row['encoder_threads'] == t) for t in (2, 4)}
    atomic_json(args.output / 'CPU_THREADS_BENCHMARK.json', result)
    print(json.dumps(result['median_pair_seconds'], indent=2))
    print('Timing benchmark only: inspect actual bitrate/detection changes before changing the official profile.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--decoder', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-indices', type=int, nargs=2, default=[0, 1])
    p.add_argument('--frames', type=int, default=256)
    p.add_argument('--bitrate-kbps', type=int, choices=[50, 80, 120, 160, 200], default=120)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--beta', type=float, default=.1)
    run(p.parse_args())
