# Paste this into ONE Kaggle cell BEFORE the failed build/hash cell.
# Requires the setup cell's WORK, REPO, CANONICAL, run, and subprocess.
# It prepares the original build directory; rerun the failed cell afterwards.
# No tracked Agent source, reward, checkpoint, or expected hash is modified.

assert str(WORK) == '/kaggle/working'
assert CANONICAL == 'e0b586c46f61e2ebbf131e4eb83f98817ff6e374'
CODEC = WORK / 'vvenc_rl_source'
assert CODEC != REPO
if not (CODEC / '.git').is_dir():
    assert not CODEC.exists(), 'Existing non-Git directory: ' + str(CODEC)
    run(['git', 'clone', '--branch', 'full-training-dev', '--depth', '60',
         'https://github.com/youngan2810/vvenc_rl.git', CODEC])
assert not subprocess.check_output(
    ['git', '-C', str(CODEC), 'status', '--porcelain', '--untracked-files=no'],
    text=True).strip(), 'Canonical checkout has tracked edits'
if subprocess.run(['git', '-C', str(CODEC), 'cat-file', '-e', CANONICAL + '^{commit}'],
                  capture_output=True).returncode:
    run(['git', '-C', CODEC, 'fetch', 'origin', CANONICAL])
run(['git', '-C', CODEC, 'checkout', '--detach', CANONICAL])
assert subprocess.check_output(
    ['git', '-C', str(CODEC), 'rev-parse', 'HEAD'], text=True).strip() == CANONICAL
print('CANONICAL_BUILD_PATH_READY', CODEC)
print('Now rerun the failed build/hash cell. Keep both SHA assertions.')
