"""CPU policy-only IPC. Teacher targets/rewards are never read here."""
import argparse
import json
from pathlib import Path
import time

import torch

from agent_v4 import STATE_VERSION
from agent_v4.checkpoint import load
from agent_v4.state import SourceFrames, StateBuilder
from ppo_policy_bridge import write_action_atomic


def main():
    p = argparse.ArgumentParser()
    for name in ("state-file", "action-dir", "trajectory-file", "checkpoint", "source-yuv", "ready-file"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--idle-timeout", type=float, default=600)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--zero-action", action="store_true", help="Codec identity gate only, never trainable")
    a = p.parse_args()
    if a.state_file.exists() or a.trajectory_file.exists() or a.ready_file.exists():
        p.error("Fresh per-episode paths required")
    if a.action_dir.exists() and any(a.action_dir.iterdir()):
        p.error("Stale action directory")
    if a.idle_timeout <= 0:
        p.error("Invalid timeout")
    torch.set_num_threads(1)
    ckpt, model, _ = load(a.checkpoint)
    model.eval()
    torch.manual_seed(a.seed)  # set AFTER model initialization/load
    builder = StateBuilder(a.frames, model.config.rich_state)
    source = SourceFrames(a.source_yuv, a.frames) if model.config.content else None
    for path in (a.state_file, a.trajectory_file, a.ready_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    a.action_dir.mkdir(parents=True, exist_ok=True)
    a.ready_file.write_text("READY\n", encoding="ascii")
    last = time.monotonic()
    while not a.state_file.is_file():
        if time.monotonic()-last > a.idle_timeout:
            raise TimeoutError("Encoder did not create state file")
        time.sleep(.02)
    with a.state_file.open(encoding="utf-8") as f, a.trajectory_file.open("w", encoding="utf-8") as out:
        pending = ""
        while len(builder.seen) < a.frames:
            pending += f.readline()
            if not pending.endswith("\n"):
                if time.monotonic()-last > a.idle_timeout:
                    raise TimeoutError(f"No complete STATE {len(builder.seen)}/{a.frames}")
                time.sleep(.02)
                continue
            line, pending = pending, ""
            last = time.monotonic()
            poc, state = builder.observe(line)
            s = {k:v[None] for k,v in state.items()}
            s["image"] = (source.thumbnail(poc)[None] if source else
                          torch.zeros(1, 1, 160, 240, dtype=torch.uint8))
            with torch.no_grad():
                action, lp, value, probs = model.sample(s, a.deterministic,
                                                       actor_only=a.deterministic or a.zero_action)
            actions = [0]*12 if a.zero_action else action[0].tolist()
            record = {"schema": STATE_VERSION, "poc": poc,
                      **{k:v.tolist() for k,v in state.items()},
                      "action": actions, "log_prob": float(lp[0]),
                      "value": None if a.deterministic or a.zero_action else float(value[0]),
                      "probabilities": probs[0].tolist(),
                      "trainable": not a.zero_action and not a.deterministic}
            # Persist state before releasing the encoder to the next picture.
            out.write(json.dumps(record, allow_nan=False, separators=(",", ":"))+"\n")
            out.flush()
            write_action_atomic(a.action_dir, poc, actions)
            builder.commit_action(actions)
    print(f"AGENT_V4_BRIDGE_PASS states={len(builder.seen)}", flush=True)

