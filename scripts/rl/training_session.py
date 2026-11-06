"""Recovery helpers for the repository-owned dual-collector trainer.

Progress is the commit record. An interrupted attempt is retained, never reused
as fresh on-policy data. Completed epochs must be validated before collecting
more data, including after restarting from an exported bundle.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

from full_train import RATES, atomic_json, checkpoint_bundle, create_schedule
from real_reward_pilot import sha256_file
from validation_objective import SELECTION_METRIC, VALIDATION_VERSION, macro_objective_metrics


def contained_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts or ':' in relative or '\\' in relative:
        raise ValueError('Unsafe relative path: ' + relative)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Path escapes run directory')
    return root / path


def new_attempt(path: Path) -> Path:
    """Choose a new output name without deleting or moving interrupted work."""
    if not path.exists():
        return path
    return path.with_name(path.stem + '_retry_' + uuid.uuid4().hex[:12] + path.suffix)


def session_limit(start: int, total: int, maximum: int) -> int:
    """A session boundary must not change the on-policy batch grouping."""
    limit = min(total, start + maximum) if maximum else total
    if limit % 325 and (limit % 325) % 2:
        limit -= 1
    return limit


def validate_progress(root: Path, progress: dict, config: dict, locksha: str) -> None:
    if progress.get('training_config') != config or progress.get('locksha') != locksha:
        raise ValueError('Incompatible experiment. Use a new output directory for changed source/PPO settings.')
    completed = progress['completed']
    plan = create_schedule(config['seed'], config['epochs'])
    if len(completed) > len(plan):
        raise ValueError('Progress exceeds schedule')
    for i, row in enumerate(completed):
        item = plan[i]
        if (row['step'], row['epoch'], row['train_index'], row['target_kbps']) != (
                i+1, item['epoch']+1, item['train_index'], item['rate']):
            raise ValueError('Progress does not match deterministic episode schedule')
    steps = []
    previous_sha = None
    for i, update in enumerate(progress['updates']):
        batch = update['episode_steps']
        if update['ppo_iteration'] != i+1 or len(batch) not in (1, 2):
            raise ValueError('Invalid PPO update sequence')
        if len({(step-1)//325 for step in batch}) != 1:
            raise ValueError('Update crosses an epoch boundary')
        if len(batch) == 1 and batch[0] % 325:
            raise ValueError('Only an epoch tail may contain a single episode')
        if previous_sha and update['collector_checkpoint_sha256'] != previous_sha:
            raise ValueError('Broken collecting-policy checkpoint chain')
        previous_sha = update['checkpoint_sha256']
        steps.extend(batch)
    if steps != list(range(1, len(completed)+1)):
        raise ValueError('Missing or repeated episodes in PPO updates')
    if previous_sha and previous_sha != progress['latest_sha256']:
        raise ValueError('Latest checkpoint is not the last committed update')
    if sha256_file(contained_path(root, progress['latest_checkpoint'])) != progress['latest_sha256']:
        raise ValueError('Latest checkpoint checksum mismatch')
    epochs = [v['epoch'] for v in progress['validations']]
    if len(epochs) != len(set(epochs)) or any(e < 1 or e > len(completed)//325 for e in epochs):
        raise ValueError('Invalid validation history')


def validate_epoch_result(result: dict, checkpoint_sha: str, locksha: str,
                          frames: int, beta: float, encoder_threads: int = 4,
                          decoder_threads: int = 4, gamma: float = .99) -> None:
    expected = {'schema': 'vvenc-ppo-full-validation/1', 'split': 'val',
                'checkpoint_sha256': checkpoint_sha, 'baseline_lock_sha256': locksha,
                'frames_per_sample': frames, 'beta': beta, 'val_videos': 7,
                'targets_kbps': list(RATES), 'policy_sampling': 'deterministic',
                'no_optimizer_update': True, 'gamma': gamma,
                'checkpoint_selection_metric': SELECTION_METRIC,
                'validation_version': VALIDATION_VERSION}
    expected.update(encoder_threads=encoder_threads, decoder_threads=decoder_threads)
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError('Validation result provenance mismatch')
    pairs = result.get('pairs', [])
    if len(pairs) != 35 or {(p['val_index'], p['target_kbps']) for p in pairs} != {
            (i, r) for i in range(7) for r in RATES}:
        raise ValueError('Validation requires exactly 7 videos x 5 rates')
    if not all(math.isfinite(result[k]) for k in ('macro_ppo_score', 'macro_vanilla_score',
            'macro_ppo_discounted_return','macro_vanilla_discounted_return',
            'macro_ppo_mean_training_reward','macro_vanilla_mean_training_reward')):
        raise ValueError('Nonfinite validation score')
    computed = macro_objective_metrics(pairs)
    if any(not math.isclose(result[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in computed.items()):
        raise ValueError('Validation summary return differs from committed pairs')


def ensure_epoch_validations(args, root: Path, progress: dict, command_runner=None) -> None:
    if not args.run_validation:
        return
    if command_runner is None:
        def command_runner(cmd, log):
            with log.open('w', encoding='utf-8') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    for epoch in range(1, len(progress['completed'])//325 + 1):
        if any(v['epoch'] == epoch for v in progress['validations']):
            continue
        update = next(u for u in progress['updates'] if u['episode_steps'][-1] == epoch*325)
        checkpoint = contained_path(root, update['checkpoint_path'])
        if sha256_file(checkpoint) != update['checkpoint_sha256']:
            raise ValueError('Missing or changed epoch-end checkpoint')
        pending = progress.get('pending_validation')
        if pending and pending['epoch'] != epoch:
            raise ValueError('Pending validation epoch mismatch')
        output = contained_path(root, pending['output']) if pending else root / f'validation_epoch_{epoch:03d}'
        summary = output / 'validation_summary.json'
        if not summary.is_file():
            # Continue a pair-level run in place. Only legacy/incomplete runs
            # without an atomic progress file need a new attempt directory.
            if not (output / 'validation_progress.json').is_file():
                output = new_attempt(output)
            progress['pending_validation'] = {'epoch': epoch, 'output': output.relative_to(root).as_posix()}
            atomic_json(root / 'progress.json', progress)
            checkpoint_bundle(root, progress)
            try:
                command_runner([
                    sys.executable, str(args.repo / 'scripts/rl/full_validate.py'),
                    '--repo', str(args.repo), '--decoder', str(args.decoder),
                    '--checkpoint', str(checkpoint), '--output', str(output),
                    '--resume-bundle-root', str(root),
                    '--frames', str(args.validation_frames), '--beta', str(args.beta),
                    '--encoder-threads', str(getattr(args, 'encoder_threads', 4)),
                    '--decoder-threads', str(getattr(args, 'decoder_threads', 4)),
                ], root / (output.name + '.log'))
            finally:
                checkpoint_bundle(root, progress)
            summary = output / 'validation_summary.json'
        result = json.loads(summary.read_text(encoding='utf-8'))
        validate_epoch_result(result, update['checkpoint_sha256'], progress['locksha'],
                              args.validation_frames, args.beta,
                              getattr(args, 'encoder_threads', 4), getattr(args, 'decoder_threads', 4))
        best = max((v['selection_value'] for v in progress['validations']), default=-float('inf'))
        if result[SELECTION_METRIC] > best:
            temp = root / 'best_validation.pt.tmp'
            shutil.copyfile(checkpoint, temp)
            temp.replace(root / 'best_validation.pt')
        atomic_json(root / 'latest_validation.json', result)
        progress['validations'].append({'epoch': epoch, 'checkpoint_sha256': update['checkpoint_sha256'],
            'selection_metric': SELECTION_METRIC, 'selection_value': result[SELECTION_METRIC],
            'macro_ppo_discounted_return': result['macro_ppo_discounted_return'],
            'macro_vanilla_discounted_return': result['macro_vanilla_discounted_return'],
            'macro_ppo_mean_training_reward': result['macro_ppo_mean_training_reward'],
            'macro_vanilla_mean_training_reward': result['macro_vanilla_mean_training_reward'],
            'macro_ppo_score': result['macro_ppo_score'], 'macro_vanilla_score': result['macro_vanilla_score']})
        progress.pop('pending_validation', None)
        atomic_json(root / 'progress.json', progress)
        checkpoint_bundle(root, progress)


def restore_bundle(archive: Path, root: Path) -> None:
    """Restore only a source-managed session bundle into a fresh output folder."""
    if root.exists() and any(root.iterdir()):
        raise ValueError('Archive restore requires an empty output directory')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 2*1024**3:
            raise ValueError('Invalid or oversized training archive')
        for name in names:
            contained_path(root, name)
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
        progress = json.loads(z.read('progress.json'))
        if progress.get('training_config', {}).get('session_version') != 3:
            raise ValueError('Legacy benchmark checkpoint: start a fresh official run with this source version')
        import hashlib
        if hashlib.sha256(z.read('checkpoint/latest.pt')).hexdigest() != progress['latest_sha256']:
            raise ValueError('Archive checkpoint checksum mismatch')
        latest = contained_path(root, progress['latest_checkpoint'])
        root.mkdir(parents=True, exist_ok=True)
        # Write progress last: it is the commit marker, not a partial restore.
        for name in names:
            if name == 'progress.json' or name.endswith('/'):
                continue
            dest = contained_path(root, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_bytes(z.read('checkpoint/latest.pt'))
        best = root / 'checkpoint/best_validation.pt'
        if best.is_file():
            shutil.copyfile(best, root / 'best_validation.pt')
        atomic_json(root / 'progress.json', progress)

