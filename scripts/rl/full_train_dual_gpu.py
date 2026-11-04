#!/usr/bin/env python3
"""Two-GPU on-policy VVenC/PPO training.

Collect two independently encoded Train episodes concurrently from *one*
identical frozen PPO checkpoint. Restrict each collector process to exactly
one GPU: CUDA_VISIBLE_DEVICES=0 or 1. Frozen YOLOv5n then runs as cuda:0
within its respective process. Wait for BOTH episodes, validate on-policy
identity, concatenate trajectories (with separate done=True boundaries), and
perform ONE PPO update on CPU. Never update while the second collector runs.

The VVenC/VVdeC binaries remain CPU executables; two GPUs parallelize
separate YOLO detection jobs, not individual VVenC encode threads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

from full_train import (RATES, TRAINING_SCHEMA, atomic_checkpoint,
                        atomic_json, checkpoint_bundle, choose_window,
                        clean_episode, create_schedule, load_trainer,
                        train_checkpoint, video_frame_capacity)
from ppo_agent import MODEL_VERSION, PPO_UPDATE_VERSION, PPOActorCritic, PPOTrainer
from training_session import (ensure_epoch_validations, new_attempt,
                              restore_bundle, session_limit, validate_progress)
from real_reward_pilot import (YOLO_REF, dataset_source, read_baseline_config,
                               read_lock_manifest, sha256_file)
from verify_real_ppo_update import (read_real_rollout,
                                    validate_unchanged_collecting_policy)

COLLECTORS = 2
EPISODES_PER_EPOCH = 65 * len(RATES)


def batch_sizes(start: int, limit: int, epoch_size: int = EPISODES_PER_EPOCH):
    """Never mix two epochs across one policy update; last batch can be 1."""
    while start < limit:
        size = min(COLLECTORS, limit - start, epoch_size - start % epoch_size)
        yield start, size
        start += size


def initialize_detector_assets(repo: Path, lock: dict) -> None:
    """Prefetch immutable YOLOv5 v7.0 assets to prevent parallel clone races.

    This runs once in the parent BEFORE launching workers; model tensors are
    NOT loaded in the parent to avoid occupying memory on either GPU.
    """
    expected = lock["config"]["detector"]
    source = repo.parent / "yolov5_v7_locked"
    if not (source / ".git").is_dir():
        subprocess.run(["git", "clone", "--branch", YOLO_REF, "--depth", "1",
                        "https://github.com/ultralytics/yolov5.git", str(source)], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected["commit"]:
        raise RuntimeError("Frozen detector Git commit != locked Baseline")
    weights = repo.parent / "yolov5n_v7_locked.pt"
    if not weights.is_file():
        temp = weights.with_suffix(".download.tmp")
        try:
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
                temp)
            if sha256_file(temp) != expected["weights_sha256"]:
                raise RuntimeError("Downloaded YOLO weights checksum mismatch")
            os.replace(temp, weights)
        finally:
            temp.unlink(missing_ok=True)
    if sha256_file(weights) != expected["weights_sha256"]:
        raise RuntimeError("YOLO weights checksum mismatch")
    print("FROZEN_YOLO_ASSETS_READY commit=" + commit, flush=True)


def spawn_parallel_collectors(jobs: list[dict], repo: Path, decoder: Path,
                              checkpoint: Path, seed: int, beta: float,
                              timeout_sec: int = 3600) -> list[dict]:
    """Collect independent stochastic on-policy rollouts on GPU 0 and GPU 1."""
    processes = []
    result = []
    pending = []
    began = time.monotonic()
    try:
        for index, job in enumerate(jobs):
            device = index
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "2"
            env["VCM_RL_WORKER_DEVICE"] = str(device)
            cmd = [
                sys.executable, "-u", str(repo / "scripts/rl/real_reward_pilot.py"),
                "--repo", str(repo), "--decoder", str(decoder),
                "--train-index", str(job["train_index"]),
                "--frames", str(job["frames"]),
                "--start-frame", str(job["start_frame"]),
                "--bitrate-kbps", str(job["rate"]),
                "--beta", str(beta),
                "--policy-checkpoint", str(checkpoint),
                "--policy-seed", str(seed + 100000 + job["step"]),
                "--out", str(job["episode"]),
            ]
            log_path = job["log"]
            handle = log_path.open("w", encoding="utf-8")
            handle.write("CUDA_VISIBLE_DEVICES=" + str(device) + "\n")
            handle.flush()
            started = time.monotonic()
            try:
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
            except BaseException:
                handle.close()
                raise
            processes.append({"proc": proc, "log": log_path, "handle": handle,
                              "gpu": device, "step": job["step"], "start": started})
            print(f"GPU_WORKER_STARTED step={job['step']} cuda_visible={device} "
                  f"pid={proc.pid} train={job['train_index']} rate={job['rate']}", flush=True)
        pending = list(processes)
        while pending:
            if time.monotonic() - began > timeout_sec:
                raise TimeoutError("Collector pair timed out; no PPO update performed")
            for task in list(pending):
                code = task["proc"].poll()
                if code is None:
                    continue
                task["elapsed_sec"] = time.monotonic() - task["start"]
                if code != 0:
                    tail = "\n".join(task["log"].read_text(errors="replace").splitlines()[-35:])
                    raise RuntimeError(f"GPU worker {task['gpu']} failed rc={code} at step "
                                       f"{task['step']}:\n{tail}")
                result.append(task)
                pending.remove(task)
                print(f"GPU_WORKER_PASS step={task['step']} gpu={task['gpu']} "
                      f"elapsed={task['elapsed_sec']:.1f}s", flush=True)
            if pending:
                time.sleep(0.3)
    finally:
        failed = len(result) != len(jobs)
        for task in processes:
            # Each collector owns encoder/bridge children in its process group.
            # Stop the group on failure, including children of an exited worker.
            if os.name == 'posix' and failed:
                try:
                    os.killpg(task['proc'].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif task["proc"].poll() is None:
                task["proc"].terminate()
            if task["proc"].poll() is None:
                try:
                    task["proc"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    task["proc"].kill()
                    task["proc"].wait()
            if os.name == 'posix' and failed:
                try:
                    os.killpg(task['proc'].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            task["handle"].close()
    return sorted(result, key=lambda x: x["step"])


def run(args) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU trainer REQUIRES two visible Kaggle T4 GPUs")
    if args.max_episodes < 0 or args.epochs < 1:
        raise ValueError("Invalid training schedule")
    if not 32 <= args.frames <= 512:
        raise ValueError("Frame count must be 32..512")
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError("Invalid reward beta")
    if not args.decoder.is_file() or not (args.repo / "bin/release-static/vvencapp").is_file():
        raise FileNotFoundError("Build vvencapp and vvdecapp before training")
    torch.set_num_threads(2)
    splits, records = read_lock_manifest(Path("/kaggle/input"))
    lock = read_baseline_config(Path("/kaggle/input"))
    if len(splits["train"]) != 65 or set(splits["train"]) & set(splits["val"] + splits["test"]):
        raise AssertionError("Train/Val/Test split violation")
    plan = create_schedule(args.seed, args.epochs)
    source_commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                                            text=True).strip()
    if len(source_commit) != 40:
        raise RuntimeError("Could not pin source commit")
    cfg = {"seed": args.seed, "epochs": args.epochs, "frames": args.frames,
           "beta": args.beta, "rates": list(RATES), "source_commit": source_commit,
           "model_version": MODEL_VERSION, "parallel_collectors": COLLECTORS,
           "collector_runner_sha256": sha256_file(Path(__file__).resolve()),
           "update_unit": "one_PPO_update_per_fresh_pair",
           "session_version": 2, "ppo_update_version": PPO_UPDATE_VERSION,
           "run_validation": args.run_validation,
           "validation_frames": args.validation_frames}
    root = args.output
    if args.resume_archive:
        restore_bundle(args.resume_archive, root)
    ckpts = root / "checkpoints"
    episodes = root / "episodes"
    ckpts.mkdir(parents=True, exist_ok=True)
    episodes.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    locksha = lock["lock_sha256"]
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if (progress.get("schema") != TRAINING_SCHEMA or
                progress.get("training_config") != cfg or
                progress.get("locksha") != locksha):
            raise ValueError("Resume configuration differs from original dual-GPU experiment")
        validate_progress(root, progress, cfg, locksha)
        checkpoint = root / progress["latest_checkpoint"]
        next_step = len(progress["completed"])
        if next_step != 0 and not any(x["step"] == next_step for x in progress["completed"]):
            raise RuntimeError("Progress has missing completed-episode entry")
    else:
        if any(ckpts.iterdir()) or any(episodes.iterdir()):
            raise RuntimeError("New training must start in a clean directory")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = PPOActorCritic()
        trainer = PPOTrainer(model, epochs=5, minibatch_size=64)
        checkpoint = ckpts / "ckpt_iter_000000.pt"
        initial_state = train_checkpoint(model, trainer, 0, cfg, locksha, {})
        initial_state["source_commit"] = source_commit
        atomic_checkpoint(checkpoint, initial_state)
        progress = {"schema": TRAINING_SCHEMA, "training_config": cfg,
                    "locksha": locksha, "completed": [], "validations": [],
                    "updates": [], "latest_checkpoint": str(checkpoint.relative_to(root)),
                    "latest_sha256": sha256_file(checkpoint)}
        atomic_json(progress_path, progress)
        checkpoint_bundle(root, progress)
        next_step = 0
    initialize_detector_assets(args.repo, lock)
    # Crucial on resume: an epoch can be committed while its Val is unfinished.
    ensure_epoch_validations(args, root, progress)
    cap = {}
    target = session_limit(next_step, len(plan), args.max_episodes)
    if args.max_episodes and next_step < target and target != len(plan):
        # Limit benchmark/control runs to complete pairs; odd epoch tail is okay.
        if args.max_episodes % 2:
            raise ValueError("--max-episodes must be even for a bounded dual-GPU benchmark")
    print(f"DUAL_GPU_TRAIN_INIT cuda_devices=2 episodes={next_step}/{len(plan)} "
          f"target={target} model={MODEL_VERSION}", flush=True)
    for first, size in batch_sizes(next_step, target):
        jobs = []
        for offset in range(size):
            step = first + offset
            item = plan[step]
            idx, rate, epoch = item["train_index"], item["rate"], item["epoch"]
            vid = splits["train"][idx]
            if idx not in cap:
                cap[idx] = video_frame_capacity(dataset_source(records, vid))
            start = choose_window(args.seed, epoch, idx, args.frames, cap[idx])
            ep = new_attempt(episodes / f"episode_{step+1:06d}_e{epoch+1}_train{idx:02d}_r{rate}")
            jobs.append({"step": step+1, "epoch": epoch, "train_index": idx,
                         "video_id": vid, "rate": rate, "frames": args.frames,
                         "start_frame": start, "episode": ep,
                         "log": new_attempt(root / f"collect_{step+1:06d}.log")})
        checkpoint_sha = sha256_file(checkpoint)
        start_t = time.monotonic()
        workers = spawn_parallel_collectors(jobs, args.repo, args.decoder,
                                            checkpoint, args.seed, args.beta,
                                            args.worker_timeout)
        worker_sec = {w["step"]: w["elapsed_sec"] for w in workers}
        _, model, trainer = load_trainer(checkpoint, cfg, locksha)
        all_rollout = []
        summaries = []
        for job in jobs:
            summary, rows, rollout_file = read_real_rollout(job["episode"])
            if (summary.get("video_id") != job["video_id"] or
                summary.get("start_frame") != job["start_frame"] or
                summary.get("frames") != args.frames or
                summary.get("target_kbps") != job["rate"] or
                summary.get("policy_checkpoint_sha256") != checkpoint_sha or
                summary.get("deterministic_policy") is not False or
                summary.get("baseline_lock_sha256") != locksha):
                raise RuntimeError("Rollout identity invalid; refusing off-policy PPO update")
            sample, parity = validate_unchanged_collecting_policy(model, rows)
            if not sample[-1].done or any(x.done for x in sample[:-1]):
                raise RuntimeError("Each independent episode must terminate GAE")
            all_rollout.extend(sample)
            summaries.append((job, summary, rollout_file, parity))
        # All data are from the same exact checkpoint. One combined PPO update.
        torch.manual_seed(args.seed + 200000 + first)
        before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        metrics = trainer.update(all_rollout)
        if (metrics["optimizer_steps"] < 1 or
                torch.equal(before_model["actor_cells.2.weight"], model.state_dict()["actor_cells.2.weight"])):
            raise RuntimeError("Actor was not updated")
        iter_next = len(progress["updates"]) + 1
        output_ckpt = new_attempt(ckpts / f"ckpt_iter_{iter_next:06d}.pt")
        new_state = train_checkpoint(model, trainer, iter_next, cfg, locksha, metrics)
        new_state["source_commit"] = source_commit
        atomic_checkpoint(output_ckpt, new_state)
        # Atomic progress written only after BOTH rollouts and checkpoint succeed.
        for job, summary, rollout_file, parity in summaries:
            progress["completed"].append({
                "step": job["step"], "epoch": job["epoch"]+1,
                "train_index": job["train_index"], "video_id": job["video_id"],
                "target_kbps": job["rate"], "start_frame": job["start_frame"],
                "frames": args.frames, "actual_kbps": summary["actual_kbps"],
                "mean_reward": summary["mean_reward"],
                "mean_f1_reward_term": summary["mean_f1_reward_term"],
                "mean_rate_penalty": summary["mean_f1_reward_term"] - summary["mean_reward"],
                "total_tp": summary["total_tp"], "total_fp": summary["total_fp"],
                "total_fn": summary["total_fn"], "policy_parity": parity,
                "delta_qp_histogram": summary["delta_qp_histogram"],
                "timings": summary["timings"],
                "worker_gpu": jobs.index(job),
                "collector_sec": worker_sec[job["step"]],
                "rollout_sha256": sha256_file(rollout_file),
                "episode_dir": str(job["episode"].relative_to(root)),
            })
        elapsed = time.monotonic() - start_t
        progress["updates"].append({
            "ppo_iteration": iter_next, "episode_steps": [job["step"] for job in jobs],
            "collector_checkpoint_sha256": checkpoint_sha,
            "checkpoint_sha256": sha256_file(output_ckpt),
            "checkpoint_path": output_ckpt.relative_to(root).as_posix(),
            "wall_seconds": elapsed, "workers": size, "ppo_metrics": metrics,
        })
        progress["latest_checkpoint"] = str(output_ckpt.relative_to(root))
        progress["latest_sha256"] = sha256_file(output_ckpt)
        atomic_json(progress_path, progress)
        checkpoint = output_ckpt
        checkpoint_bundle(root, progress)
        for job in jobs:
            clean_episode(job["episode"], args.keep_yuv)
            if not args.keep_bitstreams:
                (job["episode"] / "online" / "ppo.266").unlink(missing_ok=True)
        print(f"DUAL_GPU_PPO_UPDATE_PASS iteration={iter_next} episodes={len(jobs)} "
              f"total_frames={len(all_rollout)} elapsed={elapsed:.1f}s "
              f"reward={metrics['mean_reward']:.5f} KL={metrics['approx_kl']:.6f}", flush=True)
        ensure_epoch_validations(args, root, progress)
    print(f"DUAL_GPU_TRAIN_SESSION_PASS completed={len(progress['completed'])}/{len(plan)} "
          f"ppo_updates={len(progress['updates'])}", flush=True)
    if len(progress["completed"]) == len(plan):
        print("FULL_TRAINING_COMPLETE: inspect seven-video paired Validation before Test", flush=True)
    else:
        print("SAVE PPO_QP_FULL_TRAIN_LATEST.zip AND RESUME FOR REMAINING EPISODES", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-episodes", type=int, default=2,
                   help="2 = one dual-GPU on-policy benchmark; 0 = finish schedule")
    p.add_argument("--run-validation", action="store_true")
    p.add_argument("--validation-frames", type=int, default=256)
    p.add_argument("--worker-timeout", type=int, default=3600)
    p.add_argument("--keep-yuv", action="store_true")
    p.add_argument("--keep-bitstreams", action="store_true")
    p.add_argument("--resume-archive", type=Path,
                   help="Restore a v2 source-managed bundle into an empty --output")
    args = p.parse_args()
    if not 32 <= args.validation_frames <= 512 or args.worker_timeout < 60:
        p.error("Invalid validation frame count / worker timeout")
    run(args)


if __name__ == "__main__":
    main()
