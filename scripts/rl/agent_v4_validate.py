#!/usr/bin/env python3
"""Frozen paired seven-video Validation of any v4 candidate; no optimizer."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
import math

import torch

from agent_v4 import SCHEMA
from agent_v4.checkpoint import load
from agent_v4.codec import require_runtime_profile
from agent_v4.session import export_results
from agent_v4.train import request_validation, validate_pending
from agent_v4.runtime import verify_source
from real_reward_pilot import read_baseline_config, sha256_file
from full_train_dual_gpu import initialize_detector_assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","decoder","checkpoint","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--input-root",type=Path,default=Path("/kaggle/input"))
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--quick",action="store_true",help="4 pairs, diagnostic only")
    p.add_argument('--spatial-controls',action='store_true',help='Add measured uniform/frame-replay diagnostics; no selection')
    p.add_argument("--max-hours",type=float,default=11.5)
    a = p.parse_args()
    if not math.isfinite(a.max_hours) or a.max_hours <= 0:
        p.error("Invalid time budget")
    if not torch.cuda.is_available() or torch.cuda.device_count() < a.workers:
        p.error("Missing CUDA workers")
    ckpt,_,_ = load(a.checkpoint)
    exp = ckpt["experiment"]
    require_runtime_profile(exp)
    verify_source(a.repo,exp)
    a.seed, a.validation_frames, a.worker_timeout = exp["seed"],exp["validation_frames"],1800
    progress_path = a.output/"progress.json"
    import json
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if (progress.get("purpose") != "validation-only" or
                progress.get("experiment") != exp or
                progress.get("latest_sha256") != sha256_file(a.checkpoint) or
                progress.get("validation_kind") != ("quick" if a.quick else "full")):
            p.error("Output belongs to another experiment/candidate/validation kind")
    else:
        if a.output.exists() and any(a.output.iterdir()):
            p.error("Use an empty Validation output directory")
        frozen = a.output/"checkpoints/frozen.pt"
        frozen.parent.mkdir(parents=True)
        shutil.copy2(a.checkpoint,frozen)
        progress = {"schema":SCHEMA,"purpose":"validation-only","experiment":exp,
            "validation_kind":"quick" if a.quick else "full", "completed":[],"updates":[],"validations":[],
            "latest_checkpoint":"checkpoints/frozen.pt","latest_sha256":sha256_file(frozen)}
        request_validation(a,progress,"quick" if a.quick else "full")
    try:
        initialize_detector_assets(a.repo,read_baseline_config(a.input_root))
        validate_pending(a,progress,exp,time.monotonic()+a.max_hours*3600)
    finally:
        print(export_results(a.output))


if __name__ == "__main__":
    main()

