"""Audited IO-only resume of the genuine v3 EP471 checkpoint.

Checkpoint bytes, optimizer, RNG and history are retained. Only source/runner
provenance changes; the first new update loads the original checkpoint config.
"""
import copy
import hashlib
from pathlib import Path
import subprocess

BASE = '2fa64b6895c7e9d32013828905d3e5f6a0526de1'
LOCK = '54aa575cba547e4452b5ab03422c491cd5f103db5f66f4e0e80b1a2268315213'
CHECKPOINT = '17aced9b0449e3d27344b826b10bbf5dbde491a46397660e73d10444ddc82543'
ALLOWED = {'README.md', *('scripts/rl/' + n for n in (
    'ppo_policy_bridge.py', 'full_train_dual_gpu.py', 'bridge_io_upgrade.py',
    'test_bridge_io_upgrade.py', 'test_policy_bridge_io.py'))}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def patch_bridge(old):
    before = '        while len(observed) < a.frames:\n            line = src.readline()\n            if not line:'
    after = ('        pending = ""\n        while len(observed) < a.frames:\n'
             '            chunk = src.readline()\n            if chunk:\n                pending += chunk\n'
             '            # A regular file at its current EOF may expose an unfinished line.\n'
             '            # Preserve it until the encoder publishes the newline delimiter.\n'
             '            if not pending.endswith("\\n"):')
    anchor = '            last_activity = time.monotonic()\n            poc, globals_, spatial = parse_observation(line)'
    if old.count(before) != 1 or old.count(anchor) != 1:
        raise ValueError('Unexpected original bridge source')
    return old.replace(before, after).replace(anchor, '            line, pending = pending, ""\n' + anchor)


def patch_runner(old):
    anchor = '        progress = json.loads(progress_path.read_text())'
    loader = '        from alignment_upgrade import load_compatible_trainer'
    parser = '    args = p.parse_args()'
    if any(old.count(s) != 1 for s in (anchor, loader, parser)):
        raise ValueError('Unexpected original runner source')
    return old.replace(anchor, anchor + '\n'
        '        if getattr(args, "allow_bridge_io_upgrade", False):\n'
        '            from bridge_io_upgrade import upgrade_progress\n'
        '            progress = upgrade_progress(args.repo, root, progress, cfg, locksha)')\
        .replace(loader, '        from bridge_io_upgrade import load_compatible_trainer')\
        .replace(parser, '    p.add_argument("--allow-bridge-io-upgrade", action="store_true",\n'
            '                   help="Audited complete-line IO repair of the verified v3 EP471 run")\n' + parser)


def check_patch(repo):
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    if git(repo, 'status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError('Bridge IO upgrade requires clean committed source')
    changed = set(git(repo, 'diff', '--name-only', BASE, head).decode().splitlines())
    if not changed or not changed <= ALLOWED:
        raise ValueError('Upgrade changes files outside the IO repair: ' + str(changed))
    for name, patch in (('ppo_policy_bridge.py', patch_bridge), ('full_train_dual_gpu.py', patch_runner)):
        old = git(repo, 'show', BASE + ':scripts/rl/' + name).decode()
        if (Path(repo)/'scripts/rl'/name).read_text(encoding='utf-8') != patch(old):
            raise ValueError(name + ' differs beyond the audited IO repair')
    return head, hashlib.sha256(git(repo, 'diff', '--binary', BASE, head)).hexdigest()


def upgrade_progress(repo, root, progress, cfg, locksha):
    from full_train import atomic_json, checkpoint_bundle, load_trainer
    from training_session import validate_progress
    if progress['training_config'] == cfg:
        return progress
    original = progress['training_config']
    expected = dict(original, source_commit=cfg['source_commit'],
                    collector_runner_sha256=cfg['collector_runner_sha256'])
    if (original.get('source_commit') != BASE or cfg != expected or locksha != LOCK
            or progress['latest_sha256'] != CHECKPOINT
            or len(progress['completed']) != 471 or len(progress['updates']) != 236):
        raise ValueError('IO upgrade only accepts the verified v3 EP471 checkpoint with unchanged PPO settings')
    validate_progress(root, progress, original, locksha)
    load_trainer(Path(root)/progress['latest_checkpoint'], original, locksha)
    head, diff_sha = check_patch(repo)
    if head != cfg['source_commit']:
        raise ValueError('IO upgrade target source mismatch')
    audit = dict(schema='vvenc-bridge-io-upgrade/1', from_config=copy.deepcopy(original),
        to_config=copy.deepcopy(cfg), checkpoint_sha256=CHECKPOINT,
        at_completed_episodes=471, at_ppo_updates=236, source_diff_sha256=diff_sha,
        checkpoint_bytes_unchanged=True)
    backup = Path(root)/'progress_before_bridge_io_upgrade.json'
    if backup.exists():
        import json
        if json.loads(backup.read_text()) != progress:
            raise ValueError('Existing bridge upgrade backup differs')
    atomic_json(backup, progress)
    result = copy.deepcopy(progress)
    result['training_config'] = copy.deepcopy(cfg)
    result['bridge_io_upgrade'] = audit
    atomic_json(Path(root)/'progress.json', result)
    checkpoint_bundle(root, result)
    return result


def load_compatible_trainer(checkpoint, cfg, locksha, progress):
    from full_train import load_trainer
    from real_reward_pilot import sha256_file
    audit = progress.get('bridge_io_upgrade')
    if audit and sha256_file(checkpoint) == CHECKPOINT:
        original = audit.get('from_config', {})
        expected = dict(original, source_commit=cfg['source_commit'],
                        collector_runner_sha256=cfg['collector_runner_sha256'])
        if (audit.get('schema') != 'vvenc-bridge-io-upgrade/1'
                or audit.get('checkpoint_sha256') != CHECKPOINT or audit.get('to_config') != cfg
                or original.get('source_commit') != BASE or expected != cfg or locksha != LOCK
                or audit.get('at_completed_episodes') != 471 or audit.get('at_ppo_updates') != 236
                or audit.get('checkpoint_bytes_unchanged') is not True):
            raise ValueError('Invalid bridge IO upgrade provenance')
        return load_trainer(checkpoint, original, locksha)
    from alignment_upgrade import load_compatible_trainer as load_original
    return load_original(checkpoint, cfg, locksha, progress)
