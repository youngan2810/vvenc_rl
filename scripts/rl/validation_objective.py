"""Evaluate the SAME frame reward/decision order used by the PPO collector.

Checkpoint selection uses mean discounted episode return over fixed Val pairs.
Aggregate clip F1, final rate penalty and their score remain separate metrics.
This changes evaluation/selection, not the reward sent to PPO.
"""
import hashlib
import json
import math
from pathlib import Path

from au_rate_audit import read_au_bits
from reward_join import read_detections
from reward_metrics import calculate_frame_reward

SELECTION_METRIC = 'macro_ppo_discounted_return'
VALIDATION_VERSION = 3


def training_objective_metrics(stream, states, au_log, source_detections,
                               decoded_detections, video_id, frames, rate,
                               beta, gamma, trace_path=None):
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('Invalid discount gamma')
    order=[int(line.split()[1]) for line in Path(states).read_text().splitlines()
           if line.startswith('STATE ')]
    if len(order)!=frames or set(order)!=set(range(frames)):
        raise ValueError('Missing/duplicate policy decision POCs')
    au=read_au_bits(Path(au_log),Path(stream),frames)
    bits={r.poc:r.au_bits for r in au}
    ref=read_detections(Path(source_detections),video_id,set(range(frames)))
    rec=read_detections(Path(decoded_detections),video_id,set(range(frames)))
    cumulative=0;discount=1.;total=discounted=detection=penalty=0.;trace=[]
    for step,poc in enumerate(order,1):
        cumulative+=bits[poc]
        r=calculate_frame_reward(video_id=video_id,poc=poc,encoded_step=step,
            pseudo_gt=ref[poc],reconstructed=rec[poc],actual_frame_bits=bits[poc],
            cumulative_bits=cumulative,frame_rate=30,target_bps=rate*1000,beta=beta)
        total+=r.reward;discounted+=discount*r.reward
        detection+=r.detection_term;penalty+=r.bitrate_penalty
        trace.append(dict(step=step,poc=poc,au_bits=bits[poc],cumulative_bits=cumulative,
            detection_term=r.detection_term,rate_penalty=r.bitrate_penalty,
            reward=r.reward,discount=discount))
        discount*=gamma
    if trace_path is not None:
        Path(trace_path).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in trace),encoding='utf-8')
    return dict(mean_training_reward=total/frames,discounted_return=discounted,
        mean_frame_detection_term=detection/frames,
        mean_cumulative_rate_penalty=penalty/frames,gamma=gamma,
        decision_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())


def macro_objective_metrics(pairs):
    if not pairs:raise ValueError('No validation pairs')
    result={}
    for method in ('ppo','vanilla'):
        for field in ('mean_training_reward','discounted_return'):
            values=[row[method][field] for row in pairs]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite validation reward')
            result[f'macro_{method}_{field}']=sum(values)/len(values)
    return result
