"""Versioned objectives, POC-safe joining, and train-only teacher targets."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from agent_v4.state import cell_bounds, StateBuilder
from reward_metrics import DetectionCounts, match_one_to_one, calculate_frame_reward


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "clip-delta"
    rate_mode: str = "symmetric"
    beta: float = .1
    tolerance: float = 0.
    empty_fp: float = .25

    def __post_init__(self):
        if self.mode not in ("clip-delta", "legacy-frame") or self.rate_mode not in ("symmetric", "cap"):
            raise ValueError("Unknown reward contract")
        if any(not math.isfinite(x) or x < 0 for x in (self.beta, self.tolerance, self.empty_fp)):
            raise ValueError("Invalid reward parameters")
        if self.mode == "legacy-frame" and (self.rate_mode != "symmetric" or self.tolerance != 0):
            raise ValueError("Legacy control uses symmetric rate error without tolerance")


def score(counts: DetectionCounts, bits: int, n: int, target_bps: float,
          config: RewardConfig, fps=30.):
    if n < 1 or bits < 0 or target_bps <= 0 or not math.isfinite(target_bps):
        raise ValueError("Invalid clip score inputs")
    rate = max(1., bits*fps/n)
    error = math.log(rate/target_bps)
    penalty = (max(0., abs(error)-math.log1p(config.tolerance))
               if config.rate_mode == "symmetric" else
               max(0., error-math.log1p(config.tolerance)))
    # Empty clips earn no free positive quality. Penalize false positives
    # per picture; report this separately from ordinary clip F1.
    quality = (counts.f1 if counts.tp+counts.fn else -config.empty_fp*counts.fp/n)
    return quality-config.beta*penalty


def occupancy(boxes):
    """Exact UNION area per actual CTU / actual CTU area; never sum overlaps.

    Source frozen-detector boxes are teacher supervision, not input features,
    per-CTU F1, human ground truth, or an estimate of task sensitivity.
    """
    targets = []
    for x0, y0, x1, y1 in cell_bounds():
        clipped = [(max(x0,b.x1),max(y0,b.y1),min(x1,b.x2),min(y1,b.y2)) for b in boxes]
        clipped = [b for b in clipped if b[2] > b[0] and b[3] > b[1]]
        xs = sorted({x for b in clipped for x in (b[0], b[2])})
        area = 0.
        for a, z in zip(xs, xs[1:]):
            intervals = sorted((b[1], b[3]) for b in clipped if b[0] < z and b[2] > a)
            height, end = 0., -float("inf")
            for lo, hi in intervals:
                height += max(0., hi-max(lo, end))
                end = max(end, hi)
            area += (z-a)*height
        targets.append(area/((x1-x0)*(y1-y0)))
    return targets


def join(rows, aus, source, decoded, target_bps, config: RewardConfig, video_id):
    n = len(rows)
    pocs = [r["poc"] for r in rows]
    if not n or len(set(pocs)) != n or set(pocs) != set(range(n)):
        raise ValueError("Incomplete/duplicate policy POCs")
    au = {r.poc:r for r in aus}
    if len(aus) != n or set(au) != set(pocs) or set(source) != set(pocs) or set(decoded) != set(pocs):
        raise ValueError("AU/detection POC set mismatch")
    total = DetectionCounts(0, 0, 0)
    bits, previous, out = 0, 0., []
    for i, row in enumerate(rows, 1):
        poc = row["poc"]
        c = match_one_to_one(source[poc], decoded[poc], .5)
        total = DetectionCounts(total.tp+c.tp, total.fp+c.fp, total.fn+c.fn)
        bits += au[poc].au_bits
        current = score(total, bits, i, target_bps, config)
        if config.mode == "clip-delta":
            reward = current-previous
        else:
            reward = calculate_frame_reward(video_id=video_id, poc=poc, encoded_step=i,
                pseudo_gt=source[poc], reconstructed=decoded[poc],
                actual_frame_bits=au[poc].au_bits, cumulative_bits=bits,
                frame_rate=30., target_bps=target_bps, beta=config.beta,
                empty_reference_fp_penalty=config.empty_fp).reward
        previous = current
        out.append({**row, "video_id": video_id, "reward": reward, "done": i == n,
                    "teacher_occupancy": occupancy(source[poc]),
                    "au_bits": au[poc].au_bits, "au_output_step": au[poc].output_step,
                    "counts": asdict(c), "prefix_score": current})
    if config.mode == "clip-delta" and abs(sum(r["reward"] for r in out)-previous) > 1e-10:
        raise RuntimeError("Telescoping reward contract failed")
    return out, {**asdict(total), "f1": total.f1, "score": previous,
                 "actual_kbps": bits*30/n/1000, "bits": bits,
                 "reward_sum": sum(r["reward"] for r in out)}


def native_decision_rows(path, frames):
    """Baseline must use its OWN decision clock, particularly with gamma<1."""
    builder = StateBuilder(frames, rich=False)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        poc,_ = builder.observe(line)
        builder.commit_action([0]*12)
        rows.append({"poc":poc})
    if len(rows) != frames:
        raise ValueError("Incomplete baseline native decision sequence")
    return rows

