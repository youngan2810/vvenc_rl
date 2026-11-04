#!/usr/bin/env python3
"""Restartable on-policy PPO-QP training over 65 locked BDD100K Train videos.

Each epoch uses each Train ID once at each of the five target rates, i.e.
325 independent video-window episodes. PPO is updated only after genuine
encode/decode/frozen-YOLO reward collection. No Val/Test sample is trained on.

IMPORTANT: v2 CTU-local Actor is incompatible with the older pilot checkpoint.
Use a new output directory. Do not report a gain before paired validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from ppo_agent import MODEL_VERSION, PPOActorCritic, PPOTrainer
from real_reward_pilot import (DATASET_HINT, FPS, dataset_source,
                               read_baseline_config, read_lock_manifest, sha256_file)
from verify_real_ppo_update import (read_real_rollout,
                                    validate_unchanged_collecting_policy)

RATES = (50, 80, 120, 160, 200)
TRAINING_SCHEMA = "vvenc-ppo-full-training/1"
SOURCE_COMMIT = "5a76b04c4d958143ff5cab821196a5051458a655"


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_checkpoint(path: Path, ckpt: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def checkpoint_bundle(root: Path, progress: dict) -> Path:
    dest = root / "PPO_QP_FULL_TRAIN_LATEST.zip"
    tmp = dest.with_suffix(".zip.tmp")
    current = root / progress["latest_checkpoint"]
    if sha256_file(current) != progress["latest_sha256"]:
        raise RuntimeError("Latest checkpoint checksum changed; cannot export")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(current, arcname="checkpoint/latest.pt")
        z.write(root / "progress.json", arcname="progress.json")
        best = root / "best_validation.pt"
        if best.is_file():
            z.write(best, arcname="checkpoint/best_validation.pt")
        if (root / "latest_validation.json").is_file():
            z.write(root / "latest_validation.json", arcname="latest_validation.json")
        # Epoch-end weights must survive session export, including pending Val.
        for update in progress.get("updates", []):
            if update["episode_steps"][-1] % 325 == 0 and "checkpoint_path" in update:
                path = root / update["checkpoint_path"]
                if sha256_file(path) != update["checkpoint_sha256"]:
                    raise RuntimeError("Epoch checkpoint checksum mismatch")
                z.write(path, arcname=path.relative_to(root).as_posix())
        evidence = set(root.glob("*.log"))
        evidence.update(root.glob("validation_epoch_*/validation_summary.json"))
        for row in progress.get("completed", []):
            episode = root / row["episode_dir"]
            for name in ("summary.json", "online/au_bits.csv"):
                path = episode / name
                if path.is_file():
                    evidence.add(path)
        # Keep the latest update independently auditable without exporting all
        # raw rollouts from all epochs. Local episode folders retain all logs.
        latest_steps = set(progress.get("updates", [{}])[-1].get("episode_steps", [])) if progress.get("updates") else set()
        for row in progress.get("completed", []):
            if row["step"] in latest_steps:
                episode = root / row["episode_dir"]
                for pattern in ("*.jsonl", "*.log", "online/*.jsonl", "online/*.log", "online/states.txt"):
                    evidence.update(episode.glob(pattern))
        # Preserve evidence already restored from an earlier compact bundle.
        evidence.update(root.glob("episodes/*/summary.json"))
        for path in sorted(evidence):
            if path.is_file() and not path.is_symlink():
                z.write(path, arcname=path.relative_to(root).as_posix())
    os.replace(tmp, dest)
    return dest


def create_schedule(seed: int, epochs: int) -> list[dict]:
    """Deterministic 65 x 5 balanced schedule; shuffled within each epoch."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    plan = []
    for epoch in range(epochs):
        rng = random.Random(seed + 104729 * epoch)
        indices = list(range(65))
        rng.shuffle(indices)
        for idx in indices:
            rates = list(RATES)
            rng.shuffle(rates)
            for rate in rates:
                plan.append({"epoch": epoch, "train_index": idx, "rate": rate})
    return plan


def video_frame_capacity(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=True)
    # Leave margin for timestamp rounding and VFR last-frame differences.
    return max(0, int(float(result.stdout.strip()) * FPS) - 3)


def choose_window(seed: int, epoch: int, train_index: int, frames: int,
                  capacity: int) -> int:
    if capacity < frames:
        raise ValueError(f"Not enough frames: {capacity} < {frames}")
    max_offset = capacity - frames
    key = f"{seed}:{epoch}:{train_index}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return value % (max_offset + 1)


def train_checkpoint(model: PPOActorCritic, trainer: PPOTrainer, iteration: int,
                     config: dict, locksha: str, metrics: dict) -> dict:
    return {
        "schema": TRAINING_SCHEMA,
        "model_version": MODEL_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "ppo_iteration": iteration,
        "training_config": config,
        "baseline_lock_sha256": locksha,
        "source_commit": config.get("source_commit", SOURCE_COMMIT),
        "ppo_hparams": trainer.hyperparameters(),
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": (np.random.get_state()[0],
                            np.random.get_state()[1].tolist(),
                            int(np.random.get_state()[2]),
                            int(np.random.get_state()[3]),
                            float(np.random.get_state()[4])),
        "python_rng_state": random.getstate(),
    }


def load_trainer(checkpoint: Path, expected_config: dict, locksha: str):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (obj.get("schema") != TRAINING_SCHEMA
            or obj.get("model_version") != MODEL_VERSION
            or obj.get("training_config") != expected_config
            or obj.get("baseline_lock_sha256") != locksha):
        raise ValueError("Incompatible training checkpoint; do not mix architectures or plans")
    model = PPOActorCritic()
    model.load_state_dict(obj["model_state"], strict=True)
    trainer = PPOTrainer(model, **obj.get("ppo_hparams", {"epochs": 5, "minibatch_size": 64}))
    trainer.optimizer.load_state_dict(obj["optimizer_state"])
    return obj, model, trainer


def clean_episode(episode: Path, keep_yuv: bool) -> None:
    if keep_yuv:
        return
    for name in ("reference_8bit.yuv", "decoded_8bit.yuv", "decoded_10bit.yuv"):
        (episode / name).unlink(missing_ok=True)
    shutil.rmtree(episode / "online" / "actions", ignore_errors=True)


def run_command(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA GPU required for frozen YOLOv5n")
    if not args.decoder.is_file() or not (args.repo / "bin/release-static/vvencapp").is_file():
        raise FileNotFoundError("Build VVenC and VVdeC before launching training")
    if not 32 <= args.frames <= 512:
        raise ValueError("--frames must be between 32 and 512")
    if args.beta < 0 or not np.isfinite(args.beta):
        raise ValueError("beta must be finite and nonnegative")
    splits, records = read_lock_manifest(Path("/kaggle/input"))
    locked = read_baseline_config(Path("/kaggle/input"))
    if len(splits["train"]) != 65:
        raise RuntimeError("Incorrect Train split")
    locksha = locked["lock_sha256"]
    schedule = create_schedule(args.seed, args.epochs)
    config = {"seed": args.seed, "epochs": args.epochs, "frames": args.frames,
              "beta": args.beta, "rates": list(RATES), "source_commit": SOURCE_COMMIT,
              "model_version": MODEL_VERSION}
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "episodes").mkdir(exist_ok=True)
    progress_file = root / "progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["training_config"] != config or progress["locksha"] != locksha:
            raise RuntimeError("Cannot resume with changed experiment configuration")
        next_step = len(progress["completed"])
        ckpt_path = root / progress["latest_checkpoint"]
        if sha256_file(ckpt_path) != progress["latest_sha256"]:
            raise RuntimeError("Checkpoint hash mismatch on resume")
    else:
        if any((root / "checkpoints").iterdir()) or any((root / "episodes").iterdir()):
            raise RuntimeError("Fresh run requires empty checkpoint/episode folders")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = PPOActorCritic()
        trainer = PPOTrainer(model, epochs=5, minibatch_size=64)
        ckpt_path = root / "checkpoints/ckpt_iter_000000.pt"
        atomic_checkpoint(ckpt_path, train_checkpoint(model, trainer, 0, config, locksha, {}))
        progress = {"schema": TRAINING_SCHEMA, "training_config": config,
                    "locksha": locksha, "completed": [], "validations": [],
                    "latest_checkpoint": str(ckpt_path.relative_to(root)),
                    "latest_sha256": sha256_file(ckpt_path)}
        atomic_json(progress_file, progress)
        next_step = 0
        checkpoint_bundle(root, progress)
    capacities = {}
    limit = min(len(schedule), next_step + args.max_episodes) if args.max_episodes else len(schedule)
    for step in range(next_step, limit):
        item = schedule[step]
        idx, rate, epoch = item["train_index"], item["rate"], item["epoch"]
        vid = splits["train"][idx]
        if vid in splits["val"] or vid in splits["test"]:
            raise AssertionError("Train/Val/Test leakage")
        if idx not in capacities:
            capacities[idx] = video_frame_capacity(dataset_source(records, vid))
        start = choose_window(args.seed, epoch, idx, args.frames, capacities[idx])
        episode = root / "episodes" / f"episode_{step + 1:06d}_e{epoch + 1}_train{idx:02d}_r{rate}"
        if episode.exists():
            raise RuntimeError("Interrupted episode exists; quarantine it manually before resume: " + str(episode))
        print(f"[TRAIN {step+1}/{len(schedule)}] epoch={epoch+1}, video={idx}, "
              f"rate={rate}kbps, window={start}:{start+args.frames}", flush=True)
        run_command([
            sys.executable, str(args.repo / "scripts/rl/real_reward_pilot.py"),
            "--repo", str(args.repo), "--decoder", str(args.decoder),
            "--train-index", str(idx), "--frames", str(args.frames),
            "--start-frame", str(start), "--bitrate-kbps", str(rate),
            "--beta", str(args.beta), "--policy-checkpoint", str(ckpt_path),
            "--policy-seed", str(args.seed + 100000 + step),
            "--out", str(episode),
        ], root / f"collect_{step+1:06d}.log")
        summary, rows, rollout_file = read_real_rollout(episode)
        if (summary["video_id"] != vid or summary["start_frame"] != start
                or summary["target_kbps"] != rate or summary["frames"] != args.frames
                or summary["policy_checkpoint_sha256"] != sha256_file(ckpt_path)):
            raise RuntimeError("Collected episode provenance mismatch")
        previous, model, trainer = load_trainer(ckpt_path, config, locksha)
        rollout, parity = validate_unchanged_collecting_policy(model, rows)
        torch.manual_seed(args.seed + 200000 + step)
        metrics = trainer.update(rollout)
        if metrics["optimizer_steps"] < 1:
            raise RuntimeError("PPO update did not run")
        metrics.update(parity)
        next_checkpoint = root / "checkpoints" / f"ckpt_iter_{step + 1:06d}.pt"
        if next_checkpoint.exists():
            raise RuntimeError("Refuse to overwrite training checkpoint")
        atomic_checkpoint(next_checkpoint, train_checkpoint(
            model, trainer, step + 1, config, locksha, metrics))
        progress["completed"].append({
            "step": step + 1, "epoch": epoch + 1, "train_index": idx,
            "video_id": vid, "target_kbps": rate, "start_frame": start,
            "frames": args.frames, "actual_kbps": summary["actual_kbps"],
            "mean_reward": summary["mean_reward"], "ppo_metrics": metrics,
            "episode_dir": str(episode.relative_to(root)),
            "checkpoint_sha256": sha256_file(next_checkpoint),
            "rollout_sha256": sha256_file(rollout_file),
        })
        progress["latest_checkpoint"] = str(next_checkpoint.relative_to(root))
        progress["latest_sha256"] = sha256_file(next_checkpoint)
        atomic_json(progress_file, progress)
        ckpt_path = next_checkpoint
        checkpoint_bundle(root, progress)
        clean_episode(episode, args.keep_yuv)
        print(f"TRAIN_UPDATE_PASS iter={step+1} reward={summary['mean_reward']:.6f} "
              f"KL={metrics['approx_kl']:.6f}", flush=True)
        if (step + 1) % (65 * len(RATES)) == 0 and args.run_validation:
            val_dir = root / f"validation_epoch_{epoch+1:03d}"
            run_command([
                sys.executable, str(args.repo / "scripts/rl/full_validate.py"),
                "--repo", str(args.repo), "--decoder", str(args.decoder),
                "--checkpoint", str(ckpt_path), "--output", str(val_dir),
                "--frames", str(args.validation_frames), "--beta", str(args.beta),
            ], root / f"validation_epoch_{epoch+1:03d}.log")
            val = json.loads((val_dir / "validation_summary.json").read_text())
            score = float(val["macro_ppo_score"])
            best = max((v["macro_ppo_score"] for v in progress["validations"]), default=-float("inf"))
            progress["validations"].append({"epoch": epoch+1, "macro_ppo_score": score,
                                            "macro_vanilla_score": val["macro_vanilla_score"]})
            if score > best:
                shutil.copy2(ckpt_path, root / "best_validation.pt")
            shutil.copy2(val_dir / "validation_summary.json", root / "latest_validation.json")
            atomic_json(progress_file, progress)
            checkpoint_bundle(root, progress)
    print(f"TRAIN_RUN_COMPLETE completed={len(progress['completed'])}/{len(schedule)} "
          f"archive={root/'PPO_QP_FULL_TRAIN_LATEST.zip'}")
    if len(progress["completed"]) != len(schedule):
        print("PARTIAL TRAIN: RESUME with the same arguments; not full training completion")
    else:
        print("FULL_65_VIDEO_MULTIRATE_TRAINING_COMPLETE: inspect paired Validation before Test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--beta", type=float, default=0.1,
                   help="Provisional: must be chosen using Validation, not Test")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="0 means full schedule; positive means bounded training/resume")
    p.add_argument("--run-validation", action="store_true",
                   help="After each completed 325-episode epoch, evaluate 7 Val x 5 rates")
    p.add_argument("--validation-frames", type=int, default=256)
    p.add_argument("--keep-yuv", action="store_true")
    a = p.parse_args()
    if a.max_episodes < 0 or a.validation_frames < 32 or a.validation_frames > 512:
        p.error("Invalid training/validation episode length")
    run(a)


if __name__ == "__main__":
    main()

