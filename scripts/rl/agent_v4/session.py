"""Atomic progress, portable checksummed exports, fresh subprocess collectors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from agent_v4 import SCHEMA
from full_train import atomic_json
from real_reward_pilot import sha256_file
from training_session import contained_path, new_attempt

RESUME_NAME = "VVC_AGENT_V4_RESUME.zip"


def export_resume(root, progress):
    # Never replace the last valid bundle with internally inconsistent data.
    if json.loads((root/"progress.json").read_text()) != progress:
        raise ValueError("Only committed on-disk progress may be exported")
    expected = {progress["latest_checkpoint"]:progress["latest_sha256"]}
    if progress.get("best_checkpoint"):
        expected[progress["best_checkpoint"]] = progress["best_sha256"]
    if progress.get("pending_validation"):
        p = progress["pending_validation"]
        expected[p["checkpoint"]] = p["sha256"]
    for v in progress.get("validations", []):
        if v["kind"] == "full":
            expected[v["checkpoint"]] = v["sha256"]
    for name,digest in expected.items():
        if sha256_file(contained_path(root,name)) != digest:
            raise ValueError("Checkpoint checksum mismatch; preserving previous resume bundle: " + name)
    paths = {root/"progress.json", root/progress["latest_checkpoint"]}
    if progress.get("best_checkpoint"):
        paths.add(root/progress["best_checkpoint"])
    if progress.get("pending_validation"):
        paths.add(root/progress["pending_validation"]["checkpoint"])
    # Retain all full-validation candidates in the compact bundle.
    paths.update(root/v["checkpoint"] for v in progress.get("validations", []) if v["kind"] == "full")
    paths.update(root.glob("validation_*/progress.json"))
    manifest = {p.relative_to(root).as_posix():sha256_file(p) for p in sorted(paths)}
    dest = root/RESUME_NAME
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(p, p.relative_to(root).as_posix())
        z.writestr("bundle_manifest.json", json.dumps({"schema": SCHEMA, "files": manifest}))
    os.replace(tmp, dest)
    return dest


def restore(archive, root):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Restore requires an empty output directory")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
            raise ValueError("Duplicate entries/missing v4 bundle manifest")
        for name in names:
            contained_path(root, name)
        manifest = json.loads(z.read("bundle_manifest.json"))
        if manifest.get("schema") != SCHEMA or set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("Bundle entries differ from manifest")
        # Validate every entry before materializing anything.
        for name, digest in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Corrupt resume archive: " + name)
        root.mkdir(parents=True, exist_ok=True)
        for name in manifest["files"]:
            path = contained_path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))


def export_results(root):
    """Full evidence separate from resume; never include raw YUV or caches."""
    dest = root/"VVC_AGENT_V4_RESULTS.zip"
    tmp = dest.with_suffix(".zip.tmp")
    manifest = {}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix in (".yuv", ".tmp", ".zip", ".pt"):
                continue
            if "actions" in path.parts or "__pycache__" in path.parts:
                continue
            name = path.relative_to(root).as_posix()
            z.write(path, name)
            manifest[name] = sha256_file(path)
        z.writestr("results_manifest.json", json.dumps({"schema": SCHEMA, "files": manifest}))
    os.replace(tmp, dest)
    return dest


def run_jobs(jobs, repo, decoder, checkpoint, input_root, timeout):
    """GPU i handles job i. All jobs consume identical immutable weights."""
    procs, completed = [], []
    began = time.monotonic()
    try:
        for gpu, job in enumerate(jobs):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            cmd = [sys.executable, "-u", str(repo/"scripts/rl/agent_v4_collect.py"),
                   "--repo", str(repo), "--decoder", str(decoder), "--checkpoint", str(checkpoint),
                   "--input-root", str(input_root), "--output", str(job["output"]),
                   "--split", job["split"], "--index", str(job["index"]),
                   "--frames", str(job["frames"]), "--start", str(job["start"]),
                   "--rate", str(job["rate"]), "--seed", str(job["seed"])]
            if job["split"] == "val":
                cmd.extend(["--paired", "--deterministic"])
                if job.get("zero_gate"):
                    cmd.append("--zero-gate")
            log = job["output"].with_suffix(".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            except BaseException:
                handle.close()
                raise
            procs.append((proc, handle, log, job))
            print(f"V4_COLLECTOR_STARTED gpu={gpu} pid={proc.pid} output={job['output']}", flush=True)
        while len(completed) < len(procs):
            if time.monotonic()-began > timeout:
                raise TimeoutError("v4 collector timeout; no PPO update committed")
            for proc, handle, log, job in procs:
                if proc in completed or proc.poll() is None:
                    continue
                if proc.returncode:
                    raise RuntimeError("Collector failed: " + str(log) + "\n" + log.read_text(errors="replace")[-3000:])
                completed.append(proc)
            if len(completed) < len(procs):
                time.sleep(.2)
    finally:
        failed = len(completed) != len(procs)
        for proc, handle, _, _ in procs:
            if failed and os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.kill()
            if proc.poll() is None:
                proc.wait()
            handle.close()


def read_summary(job, checkpoint_sha, experiment):
    out = job["output"]
    summary = json.loads((out/"summary.json").read_text())
    expected = {"split":job["split"], "index":job["index"], "frames":job["frames"],
                "rate":job["rate"], "start_frame":job["start"], "seed":job["seed"],
                "policy_sha256":checkpoint_sha, "experiment":experiment,
                "deterministic":job["split"] == "val"}
    if any(summary.get(k) != v for k,v in expected.items()):
        raise ValueError("Collector identity differs from frozen job")
    if sha256_file(out/"rollout.jsonl") != summary["rollout_sha256"]:
        raise ValueError("Rollout checksum changed")
    if sha256_file(out/"reference_8bit.yuv") != summary["source_sha256"]:
        raise ValueError("Source checksum changed")
    return summary

