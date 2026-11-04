"""Small, portable pair-level validation progress; no YUV/bitstreams required."""
from __future__ import annotations

import json
import math
from pathlib import Path

from full_train import RATES, atomic_json


def pair_key(row: dict) -> tuple[int, int]:
    return row['val_index'], row['target_kbps']


def load_validation_progress(output: Path, config: dict) -> dict:
    path = output / 'validation_progress.json'
    if path.is_file():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state.get('schema') != 'vvenc-paired-validation-progress/1' or state.get('config') != config:
            raise ValueError('Cannot resume Validation with a changed checkpoint/protocol/thread count')
        keys = set()
        for row in state['completed']:
            key = pair_key(row)
            if key in keys or key[0] not in range(len(config['video_ids'])) or key[1] not in RATES:
                raise ValueError('Invalid or duplicate completed validation pair')
            if row['video_id'] != config['video_ids'][key[0]]:
                raise ValueError('Validation pair video identity mismatch')
            for method in ('vanilla', 'ppo'):
                if not all(math.isfinite(row[method][k]) for k in ('score', 'actual_kbps', 'f1')):
                    raise ValueError('Nonfinite validation pair')
            keys.add(key)
        return state
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing Validation directory has no resumable pair progress')
    output.mkdir(parents=True, exist_ok=True)
    state = {'schema': 'vvenc-paired-validation-progress/1', 'config': config,
             'completed': [], 'detector': None, 'elapsed_seconds': 0.0}
    atomic_json(path, state)
    return state


def commit_validation_pair(output: Path, state: dict, pair: dict, elapsed_seconds: float) -> None:
    if any(pair_key(row) == pair_key(pair) for row in state['completed']):
        raise ValueError('Validation pair already committed')
    state['completed'].append(pair)
    state['elapsed_seconds'] = elapsed_seconds
    atomic_json(output / 'validation_progress.json', state)
