#!/usr/bin/env python3
"""Strict, no-gradient paired Validation: 7 held-out Val videos x five rates.

This is a diagnostic on frozen YOLOv5n pseudo-label preservation, not mAP
with human ground truth. Policy and vanilla use identical preprocessing,
window, encoder settings and the same frozen detector. Never accesses Test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from ppo_agent import MODEL_VERSION
from real_reward_pilot import (
    BITRATE_KBPS, FPS, W, H, convert_10bit_to_8bit, dataset_source, detect_all,
    encode_with_ppo, load_frozen_yolov5, preprocess_mov, read_baseline_config,
    read_lock_manifest, run_cmd, sha256_file,
)
from reward_join import read_detections
from reward_metrics import match_one_to_one
from full_train import RATES, TRAINING_SCHEMA, atomic_json, checkpoint_bundle
from training_session import new_attempt
from validation_session import load_validation_progress, commit_validation_pair, pair_key


def vanilla_encode(encoder: Path, source: Path, output: Path, frames: int,
                   bitrate_kbps: int, encoder_threads: int = 4):
    if encoder_threads < 1:
        raise ValueError('encoder_threads must be positive')
    output.mkdir(parents=True, exist_ok=False)
    stream = output / "vanilla.266"
    env = dict(os.environ)
    for key in ("VVENC_RL_QP_MAP", "VVENC_RL_ACTION_DIR", "VVENC_RL_STATE_FILE", "VVENC_RL_AU_BITS_FILE"):
        env.pop(key, None)
    cmd = [str(encoder), "-i", str(source), "-s", f"{W}x{H}",
           "-r", str(FPS), "-f", str(frames), "-c", "yuv420",
           "--preset", "medium", "--internal-bitdepth", "10",
           "--bitrate", f"{bitrate_kbps}k", "--passes", "1", "--qpa", "1",
           "--intraperiod", "32", "--refreshtype", "cra",
           "--sdr", "sdr_709", "--threads", str(encoder_threads), "--mtprofile", "-1",
           "--additional", "GOPSize=32:LookAhead=1", "-o", str(stream)]
    run_cmd(cmd, output / "encoder.log", env=env, timeout=1200)
    return stream


def get_counts(source_det: Path, reconstructed_det: Path, video_id: str, frames: int):
    pocs = set(range(frames))
    source = read_detections(source_det, video_id, pocs)
    reconstructed = read_detections(reconstructed_det, video_id, pocs)
    tp = fp = fn = 0
    for i in range(frames):
        c = match_one_to_one(source[i], reconstructed[i], 0.5)
        tp += c.tp
        fp += c.fp
        fn += c.fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.0}


def evaluate_bitstream(decoder: Path, detector, stream: Path, video_id: str,
                       source_det: Path, frames: int, out: Path, bitrate: int,
                       beta: float, decoder_threads: int = 4):
    if decoder_threads < 1:
        raise ValueError('decoder_threads must be positive')
    decoded10 = out / "decoded_10bit.yuv"
    decoded8 = out / "decoded_8bit.yuv"
    run_cmd([str(decoder), "-b", str(stream), "-o", str(decoded10), "--threads", str(decoder_threads)],
            out / "decoder.log", timeout=600)
    convert_10bit_to_8bit(decoded10, decoded8, frames)
    from decoder_alignment import verify_decode_mapping
    verify_decode_mapping(stream, decoded10, decoder, frames, decoder_threads)
    det_file = out / "decoded_detections.jsonl"
    detect_all(detector, decoded8, frames, video_id, det_file)
    counts = get_counts(source_det, det_file, video_id, frames)
    from math import log
    actual_bps = stream.stat().st_size * 8 * FPS / frames
    counts.update({
        "actual_kbps": actual_bps / 1000,
        "target_kbps": bitrate,
        "rate_penalty": beta * abs(log(max(actual_bps,1.0) / (bitrate * 1000))),
    })
    counts["score"] = counts["f1"] - counts["rate_penalty"]
    for f in (decoded10, decoded8):
        f.unlink(missing_ok=True)
    return counts


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU required")
    if not 32 <= args.frames <= 512:
        raise ValueError("Val frames must be 32..512")
    if not 1 <= args.val_videos <= 7:
        raise ValueError("Val subset must be within 1..7")
    if args.encoder_threads < 1 or args.decoder_threads < 1:
        raise ValueError('Encoder/decoder threads must be positive')
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    bundle_root = getattr(args, 'resume_bundle_root', None)
    bundle_progress = None
    if bundle_root is not None:
        bundle_progress = json.loads((bundle_root / 'progress.json').read_text(encoding='utf-8'))
        pending = bundle_progress.get('pending_validation', {})
        if not pending.get('output') or (bundle_root / pending['output']).resolve() != args.output.resolve():
            raise ValueError('Validation output does not match pending training progress')
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != TRAINING_SCHEMA or ckpt.get("model_version") != MODEL_VERSION:
        raise ValueError("Only full-training v2 checkpoints are valid for paired Validation")
    splits, records = read_lock_manifest(Path("/kaggle/input"))
    locked = read_baseline_config(Path("/kaggle/input"))
    if ckpt.get("baseline_lock_sha256") != locked["lock_sha256"]:
        raise ValueError("Validation checkpoint uses different baseline protocol")
    for name in ('encoder_threads', 'decoder_threads'):
        if getattr(args, name) != ckpt.get('training_config', {}).get(name, 4):
            raise ValueError('Validation thread counts must match the training checkpoint')
    video_ids = splits['val'][:args.val_videos]
    encoder = args.repo / "bin/release-static/vvencapp"
    config = {'checkpoint_sha256': sha256_file(args.checkpoint),
              'encoder_sha256': sha256_file(encoder), 'decoder_sha256': sha256_file(args.decoder),
              'baseline_lock_sha256': locked['lock_sha256'], 'frames': args.frames,
              'beta': args.beta, 'seed': args.seed, 'video_ids': video_ids,
              'rates': list(RATES), 'model_version': MODEL_VERSION,
              'encoder_threads': args.encoder_threads, 'decoder_threads': args.decoder_threads,
              'validation_version': 2}
    state = load_validation_progress(args.output, config)
    done = {pair_key(row) for row in state['completed']}
    start_time = time.monotonic()
    prior_elapsed = state['elapsed_seconds']
    detector = None
    if len(done) < len(video_ids) * len(RATES):
        detector, provenance = load_frozen_yolov5(args.repo, locked)
        state['detector'] = provenance
        atomic_json(args.output / 'validation_progress.json', state)
    for v, vid in enumerate(video_ids):
        if vid in splits["train"] or vid in splits["test"]:
            raise AssertionError("Split leakage")
        pending_rates = [rate for rate in RATES if (v, rate) not in done]
        if not pending_rates:
            continue
        sample = args.output / f"val_{v:02d}"
        sample.mkdir(exist_ok=True)
        source = new_attempt(sample / "reference.yuv")
        preprocess_mov(dataset_source(records, vid), source, args.frames,
                       source.with_suffix('.ffmpeg.log'), start_frame=0)
        source_det = source.with_suffix('.detections.jsonl')
        detect_all(detector, source, args.frames, vid, source_det)
        for rate in pending_rates:
            pair = new_attempt(sample / f"rate_{rate}")
            pair.mkdir()
            print(f"[VALIDATION] val_index={v} target={rate}kbps", flush=True)
            vanilla_dir = pair / "vanilla"
            vanilla = vanilla_encode(encoder, source, vanilla_dir, args.frames, rate, args.encoder_threads)
            vanilla_counts = evaluate_bitstream(args.decoder, detector, vanilla, vid,
                                                source_det, args.frames, vanilla_dir, rate,
                                                args.beta, args.decoder_threads)
            ppo_dir = pair / "ppo"
            ppo, trajectory, au = encode_with_ppo(
                args.repo, encoder, source, ppo_dir, args.frames,
                policy_checkpoint=args.checkpoint, policy_seed=args.seed,
                bitrate_kbps=rate, deterministic=True, encoder_threads=args.encoder_threads)
            ppo_counts = evaluate_bitstream(args.decoder, detector, ppo, vid,
                                            source_det, args.frames, ppo_dir, rate,
                                            args.beta, args.decoder_threads)
            if len(trajectory) != args.frames or len(au) != args.frames:
                raise RuntimeError("Incomplete PPO Validation trajectory")
            result_pair = {"val_index": v, "video_id": vid, "target_kbps": rate,
                            "vanilla": vanilla_counts, "ppo": ppo_counts,
                            "delta_f1": ppo_counts["f1"]-vanilla_counts["f1"],
                            "delta_kbps": ppo_counts["actual_kbps"]-vanilla_counts["actual_kbps"]}
            commit_validation_pair(args.output, state, result_pair,
                                   prior_elapsed + time.monotonic() - start_time)
            # Make each committed pair portable even if the parent/session is
            # killed before the validation subprocess returns.
            if bundle_root is not None:
                checkpoint_bundle(bundle_root, bundle_progress)
            done.add((v, rate))
            # Keep summaries and logs; drop large decoded YUVs and intermediate bitstreams.
            for file in (vanilla, ppo):
                file.unlink(missing_ok=True)
            shutil.rmtree(ppo_dir / "actions", ignore_errors=True)
        source.unlink(missing_ok=True)
    details = sorted(state['completed'], key=pair_key)
    result = {
        "schema": "vvenc-ppo-full-validation/1",
        "split": "val", "val_videos": args.val_videos,
        "targets_kbps": list(RATES), "frames_per_sample": args.frames,
        "policy_sampling": "deterministic", "no_optimizer_update": True,
        "model_version": MODEL_VERSION, "checkpoint_sha256": sha256_file(args.checkpoint),
        "baseline_lock_sha256": locked["lock_sha256"], "detector": state['detector'],
        "encoder_threads": args.encoder_threads, "decoder_threads": args.decoder_threads,
        "beta": args.beta, "elapsed_seconds": state['elapsed_seconds'],
        "pairs": details,
        "macro_ppo_score": sum(x["ppo"]["score"] for x in details)/len(details),
        "macro_vanilla_score": sum(x["vanilla"]["score"] for x in details)/len(details),
        "macro_delta_f1": sum(x["delta_f1"] for x in details)/len(details),
        "macro_delta_kbps": sum(x["delta_kbps"] for x in details)/len(details),
    }
    atomic_json(args.output / "validation_summary.json", result)
    print("NO_UPDATE_PAIRED_VALIDATION_PASS", json.dumps({
        "pairs": len(details), "macro_delta_f1": result["macro_delta_f1"],
        "macro_delta_kbps": result["macro_delta_kbps"]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--val-videos", type=int, default=7,
                   help="Use 7 for a full Validation; smaller only for development smoke")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-threads", type=int, default=4)
    p.add_argument("--decoder-threads", type=int, default=4)
    p.add_argument("--resume-bundle-root", type=Path,
                   help="Trainer-owned run folder: refresh its compact ZIP after each Val pair")
    run(p.parse_args())


if __name__ == "__main__":
    main()

