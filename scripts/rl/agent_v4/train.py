"""QP-only v4 pilot/full schedule with strict on-policy, resume and paired Val."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import subprocess
import time

import torch

from agent_v4 import SCHEMA, MODEL_VERSION, STATE_VERSION, UPDATE_VERSION
from agent_v4.checkpoint import load, save
from agent_v4.model import Agent, ModelConfig, PPOConfig, Trainer
from agent_v4.reward import RewardConfig
from agent_v4.session import (export_resume, export_results, restore,
                              run_jobs, read_summary, new_attempt)
from agent_v4.state import SourceFrames, stack_states
from decoder_alignment import require_auditor
from full_train import atomic_json, create_schedule, choose_window, video_frame_capacity
from full_train_dual_gpu import initialize_detector_assets, batch_sizes
from real_reward_pilot import (dataset_source, read_lock_manifest,
                               read_baseline_config, sha256_file)

ENCODER_SHA = "856cee2e9f0e419423aff85c55fe8d01cfbd9b18e6847bd145bcde940080c2cb"
DECODER_SHA = "1553776eb512e02335bb4925993a6b86fecb459f06cb51045539866091d38a73"


def valid_progress(root, progress, experiment, plan):
    if (progress.get("schema") != SCHEMA or progress.get("purpose") != "training" or
            progress.get("experiment") != experiment):
        raise ValueError("Run/config mismatch; v4 cannot resume v3")
    completed = progress["completed"]
    if len(completed) > len(plan):
        raise ValueError("Progress beyond schedule")
    for i, row in enumerate(completed):
        if row["step"] != i+1 or row["plan"] != plan[i]:
            raise ValueError("Non-contiguous or altered schedule")
    steps = [n for u in progress["updates"] for n in u["steps"]]
    if steps != list(range(1, len(completed)+1)):
        raise ValueError("Update/episode chain mismatch")
    if sha256_file(root/progress["latest_checkpoint"]) != progress["latest_sha256"]:
        raise ValueError("Latest checkpoint checksum mismatch")
    ckpt, _, _ = load(root/progress["latest_checkpoint"], experiment)
    if ckpt["iteration"] != len(progress["updates"]):
        raise ValueError("Checkpoint/progress iteration mismatch")
    if progress.get("updates") and progress["updates"][-1]["checkpoint_sha256"] != progress["latest_sha256"]:
        raise ValueError("Latest update does not name current checkpoint")
    pending = progress.get("pending_validation")
    if pending and sha256_file(root/pending["checkpoint"]) != pending["sha256"]:
        raise ValueError("Pending Validation checkpoint changed")


def commit_progress(root, progress):
    atomic_json(root/"progress.json", progress)
    export_resume(root, progress)


def cleanup_raw(output):
    # Keep trajectories, detections, bitstreams, AU/POC audits, logs and metrics.
    for path in output.rglob("*.yuv"):
        path.unlink()


def validate_pending(a, progress, experiment, deadline):
    pending = progress.get("pending_validation")
    if pending is None:
        return True
    ckpt = a.output/pending["checkpoint"]
    kind = pending["kind"]
    nvideos = 7 if kind == "full" else 2
    rates = [50,80,120,160,200] if kind == "full" else [50,120]
    out = a.output/pending["output"]
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"checkpoint_sha256":pending["sha256"], "experiment":experiment,
           "kind":kind, "videos":nvideos, "rates":rates, "frames":a.validation_frames}
    state_file = out/"progress.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"config":cfg, "completed":[]}
    if state["config"] != cfg:
        raise ValueError("Pending Validation configuration differs")
    done = {(s["index"],s["rate"]) for s in state["completed"]}
    if len(done) != len(state["completed"]) or not done <= {(i,r) for i in range(nvideos) for r in rates}:
        raise ValueError("Corrupt Validation progress")
    remaining = [(i, r) for i in range(nvideos) for r in rates if (i,r) not in done]
    for first in range(0, len(remaining), a.workers):
        if time.monotonic()+a.worker_timeout+600 >= deadline:
            commit_progress(a.output, progress)
            return False
        jobs = [{"split":"val", "index":i, "rate":r, "frames":a.validation_frames,
                 "start":0, "seed":a.seed, "zero_gate":first == 0 and offset == 0,
                 "output":new_attempt(out/f"val{i:02d}_r{r}")}
                for offset,(i,r) in enumerate(remaining[first:first+a.workers])]
        run_jobs(jobs, a.repo, a.decoder, ckpt, a.input_root, a.worker_timeout)
        for job in jobs:
            summary = read_summary(job, pending["sha256"], experiment)
            state["completed"].append(summary)
            atomic_json(state_file, state)
            # Save each completed paired result before starting another batch.
            export_resume(a.output, progress)
            cleanup_raw(job["output"])
    summaries = state["completed"]
    metrics = {"kind":kind, "checkpoint":pending["checkpoint"], "sha256":pending["sha256"],
               "episodes":pending["episodes"], "pairs":len(summaries),
               "mean_score":sum(s["metrics"]["score"] for s in summaries)/len(summaries),
               "mean_selection_score":sum(s["metrics"]["selection_score"] for s in summaries)/len(summaries),
               "macro_delta_score":sum(s["delta_score"] for s in summaries)/len(summaries),
               "macro_delta_f1":sum(s["delta_f1"] for s in summaries)/len(summaries),
               "macro_delta_kbps":sum(s["delta_kbps"] for s in summaries)/len(summaries)}
    atomic_json(out/"summary.json", metrics)
    progress["validations"].append(metrics)
    # Only compare identical full 7x5 validation protocols. Quick probes do
    # not compete for best_validation, and never stand in for final Test.
    if kind == "full" and ("best_score" not in progress or metrics["mean_selection_score"] > progress["best_score"]):
        progress.update(best_score=metrics["mean_selection_score"], best_checkpoint=pending["checkpoint"],
                        best_sha256=pending["sha256"])
    del progress["pending_validation"]
    commit_progress(a.output, progress)
    print("V4_PAIRED_VALIDATION_PASS " + json.dumps(metrics), flush=True)
    return True


def request_validation(a, progress, kind):
    n = len(progress["completed"])
    progress["pending_validation"] = {
        "kind":kind, "episodes":n, "checkpoint":progress["latest_checkpoint"],
        "sha256":progress["latest_sha256"],
        "output":f"validation_{n:06d}_{kind}"}
    commit_progress(a.output, progress)


def run(a):
    began = time.monotonic()
    deadline = began+a.max_hours*3600
    if not torch.cuda.is_available() or torch.cuda.device_count() < a.workers:
        raise RuntimeError(f"Requires {a.workers} visible CUDA GPUs for frozen-detector workers")
    if not 32 <= a.frames <= 512 or not 32 <= a.validation_frames <= 512:
        raise ValueError("frames/validation-frames must be32..512")
    if a.epochs < 1 or a.max_episodes < 0 or a.val_every_updates < 1 or a.worker_timeout < 60:
        raise ValueError("Invalid schedule/timeout")
    torch.set_num_threads(2)
    encoder = a.repo/"bin/release-static/vvencapp"
    auditor = require_auditor(a.decoder)
    if sha256_file(encoder) != ENCODER_SHA or sha256_file(a.decoder) != DECODER_SHA:
        raise ValueError("Build the original pinned Linux codec profile; binary hash mismatch")
    source_commit = subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"], text=True).strip()
    if subprocess.check_output(["git","-C",str(a.repo),"status","--porcelain","--untracked-files=no"], text=True).strip():
        raise ValueError("Training source must be committed and clean")
    splits, records = read_lock_manifest(a.input_root)
    lock = read_baseline_config(a.input_root)
    presets = {"categorical-control":ModelConfig(False,False,False),
               "encoder-history":ModelConfig(True,False,True),
               "content-task":ModelConfig(True,True,True)}
    mc = presets[a.preset]
    reward_mode = a.reward or ("legacy-frame" if a.preset == "categorical-control" else "clip-delta")
    rc = RewardConfig(reward_mode, a.rate_mode, a.beta, a.rate_tolerance)
    pc = PPOConfig(gamma=1. if reward_mode == "clip-delta" else .99,
                   gae_lambda=1. if reward_mode == "clip-delta" else .95,
                   auxiliary=a.auxiliary if a.auxiliary is not None else (.1 if mc.content else 0.))
    experiment = {"model_version":MODEL_VERSION, "state_version":STATE_VERSION,
        "model_config":asdict(mc), "ppo_config":asdict(pc), "reward":asdict(rc), "update_version":UPDATE_VERSION,
        "preset":a.preset, "source_commit":source_commit,
        "encoder_sha256":sha256_file(encoder), "decoder_sha256":sha256_file(a.decoder),
        "auditor_sha256":sha256_file(auditor), "baseline_lock_sha256":lock["lock_sha256"],
        "seed":a.seed, "epochs":a.epochs, "frames":a.frames,
        "validation_frames":a.validation_frames, "workers":a.workers,
        "encoder_threads":4, "decoder_threads":4, "val_every_updates":a.val_every_updates,
        "schedule_version":"interleaved-65-videos-v2", "split_ids_sha256":
        __import__("hashlib").sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()}
    plan = create_schedule(a.seed, a.epochs)
    if a.resume_archive:
        restore(a.resume_archive, a.output)
    progress_path = a.output/"progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        valid_progress(a.output, progress, experiment, plan)
    else:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("New v4 run requires an empty output; never use the EP650 directory")
        torch.manual_seed(a.seed)
        model = Agent(mc)
        trainer = Trainer(model, pc)
        initial = a.output/"checkpoints/iter_000000.pt"
        save(initial, model, trainer, experiment, 0)
        progress = {"schema":SCHEMA, "purpose":"training", "experiment":experiment, "completed":[], "updates":[],
                    "validations":[], "latest_checkpoint":initial.relative_to(a.output).as_posix(),
                    "latest_sha256":sha256_file(initial)}
        commit_progress(a.output, progress)
    initialize_detector_assets(a.repo, lock)
    try:
        n = len(progress["completed"])
        # Recover an interruption between the PPO commit and Val scheduling.
        # A fresh run also proves byte identity before its first PPO update.
        if (not progress.get("pending_validation") and
                (n == 0 or n%325 == 0) and
                not any(v["episodes"] == n and v["kind"] == ("quick" if n == 0 else "full")
                        for v in progress["validations"])):
            request_validation(a, progress, "quick" if n == 0 else "full")
        if not validate_pending(a, progress, experiment, deadline):
            return
        start = len(progress["completed"])
        limit = min(len(plan), start+a.max_episodes) if a.max_episodes else len(plan)
        print(f"V4_TRAIN_INIT completed={start} session_target={limit} total={len(plan)} "
              f"new_run={start == 0} model={MODEL_VERSION}", flush=True)
        capacity = {}
        # Keep each PPO batch inside an epoch. One final worker is valid.
        first = start
        while first < limit:
            if time.monotonic()+a.worker_timeout+600 >= deadline:
                break
            size = min(a.workers, limit-first, 325-first%325)
            checkpoint = a.output/progress["latest_checkpoint"]
            ckpt_sha = sha256_file(checkpoint)
            jobs = []
            for step in range(first, first+size):
                item = plan[step]
                idx = item["train_index"]
                if idx not in capacity:
                    capacity[idx] = video_frame_capacity(dataset_source(records, splits["train"][idx]))
                offset = choose_window(a.seed, item["epoch"], idx, a.frames, capacity[idx])
                jobs.append({"split":"train", "index":idx, "rate":item["rate"],
                    "frames":a.frames, "start":offset, "seed":a.seed+100000+step,
                    "output":new_attempt(a.output/"episodes"/f"episode_{step+1:06d}")})
            run_jobs(jobs, a.repo, a.decoder, checkpoint, a.input_root, a.worker_timeout)
            if sha256_file(checkpoint) != ckpt_sha:
                raise ValueError("Collecting checkpoint changed")
            ckpt, model, trainer = load(checkpoint, experiment, train=True)
            batches, summaries = [], []
            for job in jobs:
                summary = read_summary(job, ckpt_sha, experiment)
                if summary["video_id"] != splits["train"][job["index"]]:
                    raise ValueError("Train video identity changed")
                rows = [json.loads(s) for s in (job["output"]/"rollout.jsonl").read_text().splitlines()]
                if len(rows) != a.frames or any(not r["trainable"] for r in rows):
                    raise ValueError("Only fresh stochastic training data accepted")
                if {r["poc"] for r in rows} != set(range(a.frames)) or len({r["poc"] for r in rows}) != len(rows):
                    raise ValueError("Missing or duplicate rollout POC")
                if not rows[-1]["done"] or any(r["done"] for r in rows[:-1]):
                    raise ValueError("Incorrect GAE episode boundary")
                if mc.content:
                    with SourceFrames(job["output"]/"reference_8bit.yuv",a.frames) as source:
                        collected_state = stack_states(rows,source,True)
                else:
                    collected_state = stack_states(rows,None,False)
                batches.append((collected_state,rows))
                summaries.append(summary)
            state = {k:torch.cat([b[0][k] for b in batches]) for k in batches[0][0]}
            rows = [r for _,rr in batches for r in rr]
            actions = torch.tensor([r["action"] for r in rows], dtype=torch.int64)
            old_lp = torch.tensor([r["log_prob"] for r in rows])
            old_value = torch.tensor([r["value"] for r in rows])
            rewards = torch.tensor([r["reward"] for r in rows])
            dones = torch.tensor([r["done"] for r in rows], dtype=torch.bool)
            targets = torch.tensor([r["teacher_occupancy"] for r in rows])
            metrics = trainer.update(state, actions, old_lp, old_value, rewards, dones, targets)
            metrics["feature_audit"] = {
                "local_channel_mean":state["local"].mean((0,2,3)).tolist(),
                "local_channel_std":state["local"].std((0,2,3),unbiased=False).tolist(),
                "global_std":state["global"].std(0,unbiased=False).tolist(),
                "history_valid_fraction":float(state["history"][:,:,-1].mean())}
            iteration = len(progress["updates"])+1
            updated = new_attempt(a.output/"checkpoints"/f"iter_{iteration:06d}.pt")
            save(updated, model, trainer, experiment, iteration)
            for j, summary in enumerate(summaries):
                step = first+j
                progress["completed"].append({"step":step+1, "plan":plan[step],
                    "episode":jobs[j]["output"].relative_to(a.output).as_posix(),
                    "metrics":summary["metrics"], "source_sha256":summary["source_sha256"],
                    "rollout_sha256":summary["rollout_sha256"], "action_histogram":summary["action_histogram"]})
            progress["updates"].append({"iteration":iteration, "steps":list(range(first+1,first+size+1)),
                "collecting_policy_sha256":ckpt_sha, "checkpoint_sha256":sha256_file(updated),
                "metrics":metrics, "transitions":len(rows)})
            progress.update(latest_checkpoint=updated.relative_to(a.output).as_posix(),
                            latest_sha256=sha256_file(updated))
            commit_progress(a.output, progress)  # sole commit point after both rollouts + PPO + checkpoint
            for job in jobs:
                cleanup_raw(job["output"])
            first += size
            print(f"V4_PPO_UPDATE_PASS iteration={iteration} episodes={first}/{len(plan)} "
                  f"frames={len(rows)} KL={metrics['exact_final_kl']:.6f}", flush=True)
            kind = "full" if first%325 == 0 else (
                "quick" if iteration%a.val_every_updates == 0 or first == limit else None)
            if kind:
                request_validation(a, progress, kind)
                if not validate_pending(a, progress, experiment, deadline):
                    break
        print(f"V4_SESSION_END completed={len(progress['completed'])}/{len(plan)} "
              f"best_full_validation={progress.get('best_checkpoint')}", flush=True)
    finally:
        # On ordinary errors/SIGTERM/interrupt preserve all available evidence.
        # A hard VM kill can only guarantee the last atomic resume bundle.
        # Only export the committed on-disk state. An exception halfway
        # through a progress write must not publish an in-memory new head.
        progress = json.loads(progress_path.read_text())
        export_resume(a.output, progress)
        print("V4_RESULTS=" + str(export_results(a.output)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "decoder", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--resume-archive", type=Path)
    p.add_argument("--preset", choices=("categorical-control","encoder-history","content-task"), default="content-task")
    p.add_argument("--reward", choices=("clip-delta","legacy-frame"))
    p.add_argument("--rate-mode", choices=("symmetric","cap"), default="symmetric")
    p.add_argument("--rate-tolerance", type=float, default=0.)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--auxiliary", type=float)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--validation-frames", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-episodes", type=int, default=10, help="ADDITIONAL episodes;0 finishes the locked schedule")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--workers", type=int, choices=(1,2), default=2)
    p.add_argument("--val-every-updates", type=int, default=25)
    p.add_argument("--worker-timeout", type=int, default=1800)
    p.add_argument("--max-hours", type=float, default=11.5)
    a = p.parse_args()
    if not math.isfinite(a.max_hours) or a.max_hours <= 0 or a.seed < 0:
        p.error("Invalid time budget/seed")
    if a.auxiliary is not None and (not math.isfinite(a.auxiliary) or a.auxiliary < 0):
        p.error("Invalid auxiliary weight")
    def terminate(signum, frame):
        raise KeyboardInterrupt("Session termination requested")
    signal.signal(signal.SIGTERM, terminate)
    run(a)

