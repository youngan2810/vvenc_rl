"""Reuse audited codec settings, POC transport, AU accounting and decoder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from au_rate_audit import read_au_bits
from real_reward_pilot import run_cmd, W, H, FPS


def encode(repo, encoder, source, out, frames, checkpoint, rate, seed=7,
           deterministic=False, threads=4, zero=False):
    out.mkdir(parents=True, exist_ok=False)
    action_dir, states, trajectory, ready = (out/"actions", out/"states.txt",
                                            out/"trajectory.jsonl", out/"ready.txt")
    cmd = [sys.executable, "-u", str(repo/"scripts/rl/agent_v4_bridge.py"),
           "--state-file", str(states), "--action-dir", str(action_dir),
           "--trajectory-file", str(trajectory), "--frames", str(frames),
           "--checkpoint", str(checkpoint), "--source-yuv", str(source),
           "--seed", str(seed), "--ready-file", str(ready)]
    if deterministic:
        cmd.append("--deterministic")
    if zero:
        cmd.append("--zero-action")
    with (out/"bridge.log").open("w", encoding="utf-8") as log:
        bridge = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+60
            while not ready.exists() and bridge.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            if not ready.exists():
                raise RuntimeError("v4 bridge did not become ready; inspect " + str(out/"bridge.log"))
            env = {k:v for k,v in os.environ.items() if not k.startswith("VVENC_RL_")}
            env.update(VVENC_RL_ACTION_DIR=str(action_dir), VVENC_RL_STATE_FILE=str(states),
                       VVENC_RL_AU_BITS_FILE=str(out/"au_bits.csv"))
            stream = out/"ppo.266"
            run_cmd([str(encoder), "-i", str(source), "-s", f"{W}x{H}",
                "-r", str(FPS), "-f", str(frames), "-c", "yuv420",
                "--preset", "medium", "--internal-bitdepth", "10", "--bitrate", f"{rate}k",
                "--passes", "1", "--qpa", "1", "--intraperiod", "32", "--refreshtype", "cra",
                "--sdr", "sdr_709", "--threads", str(threads), "--mtprofile", "-1",
                "--additional", "GOPSize=32:LookAhead=1", "-o", str(stream)],
                out/"encoder.log", env=env, timeout=1800)
            if bridge.wait(timeout=60):
                raise RuntimeError("v4 bridge failed: " + (out/"bridge.log").read_text(errors="replace")[-2000:])
            rows = [json.loads(s) for s in trajectory.read_text().splitlines() if s]
            if len(rows) != frames or {r["poc"] for r in rows} != set(range(frames)):
                raise ValueError("Incomplete policy trajectory")
            return stream, rows, read_au_bits(out/"au_bits.csv", stream, frames)
        finally:
            if bridge.poll() is None:
                bridge.kill()
                bridge.wait()

