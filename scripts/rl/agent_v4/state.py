"""Causal adapter on the audited native STATE transport (unchanged codec).

Only current source POC is read. History is in *decision* order, not POC
order. No final bits/detections of the current picture enter the policy.
Fixed physical transforms avoid fitting any scaler on Validation or Test.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import math

import numpy as np
import torch

from ppo_policy_bridge import parse_observation

WIDTH, HEIGHT, FRAME_BYTES = 480, 320, 480 * 320 * 3 // 2
HISTORY, GLOBAL_DIM, LOCAL_DIM, HISTORY_DIM = 4, 16, 6, 21


def cell_bounds():
    return [(x, y, min(x + 128, WIDTH), min(y + 128, HEIGHT))
            for y in range(0, HEIGHT, 128) for x in range(0, WIDTH, 128)]


def geometry():
    return torch.tensor([[(x0+x1)/WIDTH-1, (y0+y1)/HEIGHT-1,
                          (x1-x0)*(y1-y0)/(128*128)]
                         for x0, y0, x1, y1 in cell_bounds()], dtype=torch.float32)


class SourceFrames:
    """Canonical YUV420p8 source, not VVenC's possibly MCTF-filtered buffer."""
    def __init__(self, path: Path, frames: int):
        if frames < 1 or path.stat().st_size != frames * FRAME_BYTES:
            raise ValueError("Source YUV size/format mismatch")
        self.frames = frames
        self.data = np.memmap(path, mode="r", dtype=np.uint8,
                              shape=(frames, FRAME_BYTES))

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def thumbnail(self, poc: int):
        if self.data is None:
            raise ValueError("Source frames already closed")
        if not 0 <= poc < self.frames:
            raise ValueError("Unexpected source POC")
        y = self.data[poc, :WIDTH*HEIGHT].reshape(HEIGHT, WIDTH)
        # Exact 2x2 box reduction; uint16 avoids overflow. Store uint8 in
        # rollouts, normalize once inside the model. No lookahead picture.
        reduced = y.astype(np.uint16).reshape(160, 2, 240, 2).sum((1, 3))
        return torch.from_numpy(((reduced + 2)//4).astype(np.uint8)).unsqueeze(0)


class StateBuilder:
    def __init__(self, frames: int, rich: bool = True):
        if frames < 1:
            raise ValueError("frames must be positive")
        self.frames, self.rich = frames, rich
        self.history = deque(maxlen=HISTORY)
        self.previous_native = None
        self.previous_action = None
        self.seen = set()

    def observe(self, line: str):
        if hasattr(self, "pending"):
            raise RuntimeError("Previous observation has no committed action")
        poc, g, s = parse_observation(line)
        g, s = g[0], s[0]
        if poc in self.seen or not 0 <= poc < self.frames:
            raise ValueError("Duplicate/out-of-range observation POC")
        if abs(float(g[1]) - len(self.seen)/self.frames) > 2e-6:
            raise ValueError("Native decision clock does not match episode")
        self.seen.add(poc)
        if not self.rich:
            # Controlled categorical ablation: exact legacy 44 inputs,
            # zero extras/history, same model topology.
            global_ = torch.cat((g, torch.zeros(8)))
            local = torch.cat((s, torch.zeros(3, 3, 4)))
            hist = torch.zeros(HISTORY, HISTORY_DIM)
        else:
            global_ = torch.zeros(GLOBAL_DIM)
            global_[:8] = g
            global_[0] = math.log(max(float(g[0]), 1e-6)*200/120)
            global_[4:6] = (g[4:6]*63 - 32)/16
            progress, used = float(g[1]), float(g[2])
            prev_used = float(self.previous_native[0][2]) if self.previous_native else 0.
            global_[8:] = torch.tensor([
                poc/self.frames, poc/self.frames-progress, used-progress,
                math.log1p(self.frames)/math.log(513),
                used-prev_used, float(self.previous_native is not None),
                (poc % 32)/32, float(self.frames <= 32)])
            local = torch.zeros(LOCAL_DIM, 3, 4)
            local[:2] = torch.log1p(s[:2].clamp_min(0)*100)/math.log(101)
            local[2] = (s[2]*63-32)/16
            if self.previous_native is not None:
                local[3] = self.previous_action.reshape(3, 4).float()/2
                local[4] = (s[2]-self.previous_native[1][2])*63/16
                local[5] = 1  # previous decision available, not previous display frame
            hist = torch.zeros(HISTORY, HISTORY_DIM)
            if self.history:
                hist[-len(self.history):] = torch.stack(list(self.history))
        if not all(torch.isfinite(t).all() for t in (global_, local, hist)):
            raise ValueError("Non-finite adapted state")
        self.pending = (g.clone(), s.clone(), global_[:8].clone())
        return poc, {"local": local, "global": global_, "history": hist}

    def commit_action(self, action):
        action = torch.as_tensor(action)
        if (action.shape != (12,) or not torch.isfinite(action).all() or
                not torch.all((action >= -2) & (action <= 2)) or
                not torch.equal(action, action.round())):
            raise ValueError("Expected 12 legal delta-QPs")
        if not hasattr(self, "pending"):
            raise RuntimeError("Action without observation")
        g, s, transformed_g = self.pending
        self.history.append(torch.cat((transformed_g, action.float()/2, torch.ones(1))))
        self.previous_native, self.previous_action = (g, s), action.clone()
        del self.pending


def stack_states(rows, source: SourceFrames | None, use_content: bool):
    if use_content and source is None:
        raise ValueError("Content-enabled rollout requires the actual source frames")
    state = {k: torch.tensor([r[k] for r in rows], dtype=torch.float32)
             for k in ("local", "global", "history")}
    state["image"] = (torch.stack([source.thumbnail(r["poc"]) for r in rows])
                      if use_content and source is not None else
                      torch.zeros(len(rows), 1, 160, 240, dtype=torch.uint8))
    return state

