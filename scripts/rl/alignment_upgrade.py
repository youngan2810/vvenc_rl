"""Explicit, checkpoint-preserving upgrade from the ec6 alignment heuristic.

Original weights/config are never rewritten. The old collecting checkpoint is
loaded with its original config once; subsequent checkpoints use the new config.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'ec6d475c9780f07acc7ac999ce48974f895bb3df'
LOCK = '54aa575cba547e4452b5ab03422c491cd5f103db5f66f4e0e80b1a2268315213'
CHECKPOINT = '96affafcbb6e495dbe4d6f94d046900acdf08b22f40640eb8ded47f241b6a115'
ALLOWED = {'README.md', *('scripts/rl/' + n for n in (
    'real_reward_pilot.py', 'full_validate.py', 'full_train_dual_gpu.py',
    'alignment_upgrade.py', 'decoder_alignment.py', 'build_vvdec_audit.py',
    'test_decoder_alignment.py', 'test_alignment_upgrade.py', 'test_codec_threads.py'))}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def check_patch(repo):
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    if git(repo, 'status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError('Upgrade requires a clean committed source checkout')
    changed = set(git(repo, 'diff', '--name-only', BASE, head).decode().splitlines())
    if not changed or not changed <= ALLOWED:
        raise ValueError('Source upgrade includes unapproved algorithm/settings files: ' + str(changed))
    # Existing collector and validation files must be EXACTLY the audited edits.
    old = git(repo, 'show', BASE+':scripts/rl/full_train_dual_gpu.py').decode()
    expected = patch_runner(old)
    if (Path(repo)/'scripts/rl/full_train_dual_gpu.py').read_text(encoding='utf-8') != expected:
        raise ValueError('Runner differs beyond the audited resume integration')
    old = git(repo, 'show', BASE+':scripts/rl/real_reward_pilot.py').decode()
    a = old.index('def temporal_alignment_check(')
    b = old.index('\ndef rgb_batches_yuv8', a)
    expected = old[:a]+'from decoder_alignment import temporal_alignment_check\n\n'+old[b:]
    expected = expected.replace('alignment = temporal_alignment_check(source_yuv, decode, args.frames)',
        'alignment = temporal_alignment_check(source_yuv, decode, args.frames,\n'
        '        bitstream=stream, decoder=args.decoder, decoder_threads=args.decoder_threads)')
    if (Path(repo)/'scripts/rl/real_reward_pilot.py').read_text(encoding='utf-8') != expected:
        raise ValueError('Collector differs beyond alignment QA')
    old = git(repo, 'show', BASE+':scripts/rl/full_validate.py').decode()
    expected = old.replace('    convert_10bit_to_8bit(decoded10, decoded8, frames)',
        '    convert_10bit_to_8bit(decoded10, decoded8, frames)\n'
        '    from decoder_alignment import verify_decode_mapping\n'
        '    verify_decode_mapping(stream, decoded10, decoder, frames, decoder_threads)')
    if (Path(repo)/'scripts/rl/full_validate.py').read_text(encoding='utf-8') != expected:
        raise ValueError('Validation differs beyond alignment QA')
    diff = git(repo, 'diff', '--binary', BASE, head)
    return head, hashlib.sha256(diff).hexdigest()


def patch_runner(old):
    return old.replace('    torch.set_num_threads(2)',
        '    from decoder_alignment import require_auditor\n'
        '    require_auditor(args.decoder)\n'
        '    torch.set_num_threads(2)')\
        .replace('        progress = json.loads(progress_path.read_text())',
        '        progress = json.loads(progress_path.read_text())\n'
        '        if getattr(args, "allow_alignment_upgrade", False):\n'
        '            from alignment_upgrade import upgrade_progress\n'
        '            progress = upgrade_progress(args.repo, root, progress, cfg, locksha)')\
        .replace('        _, model, trainer = load_trainer(checkpoint, cfg, locksha)',
        '        from alignment_upgrade import load_compatible_trainer\n'
        '        _, model, trainer = load_compatible_trainer(checkpoint, cfg, locksha, progress)')\
        .replace('    args = p.parse_args()',
        '    p.add_argument("--allow-alignment-upgrade", action="store_true",\n'
        '                   help="Explicit audited ec6 EP58 alignment-only upgrade")\n'
        '    args = p.parse_args()')


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
            or len(progress['completed']) != 58 or len(progress['updates']) != 29):
        raise ValueError('Upgrade is restricted to the verified official ec6 EP58 checkpoint/config')
    validate_progress(root, progress, original, locksha)
    load_trainer(Path(root)/progress['latest_checkpoint'], original, locksha)
    head, diff_sha = check_patch(repo)
    if head != cfg['source_commit']:
        raise ValueError('Target source commit mismatch')
    audit = dict(schema='vvenc-alignment-upgrade/1', from_config=original,
                 to_config=cfg, checkpoint_sha256=CHECKPOINT,
                 at_completed_episodes=58, at_ppo_updates=29,
                 source_diff_sha256=diff_sha, checkpoint_bytes_unchanged=True)
    # Backup is retained even if the process stops before progress commits.
    backup = Path(root)/'progress_before_alignment_upgrade.json'
    if backup.exists() and json.loads(backup.read_text()) != progress:
        raise ValueError('Existing upgrade backup differs')
    atomic_json(backup, progress)
    result = copy.deepcopy(progress)
    result['training_config'] = cfg
    result['alignment_upgrade'] = audit
    atomic_json(Path(root)/'progress.json', result)
    checkpoint_bundle(root, result)
    return result


def load_compatible_trainer(checkpoint, cfg, locksha, progress):
    from full_train import load_trainer
    from real_reward_pilot import sha256_file
    audit = progress.get('alignment_upgrade')
    if audit and sha256_file(checkpoint) == CHECKPOINT:
        if (audit.get('schema') != 'vvenc-alignment-upgrade/1'
                or audit.get('checkpoint_sha256') != CHECKPOINT
                or audit.get('to_config') != cfg
                or audit.get('from_config', {}).get('source_commit') != BASE):
            raise ValueError('Invalid alignment upgrade provenance')
        return load_trainer(checkpoint, audit['from_config'], locksha)
    return load_trainer(checkpoint, cfg, locksha)
