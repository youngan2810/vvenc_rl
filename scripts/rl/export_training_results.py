"""Export analysis evidence separately from the small per-update resume ZIP."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile


def export_analysis(root: Path) -> Path:
    if not (root / 'progress.json').is_file():
        raise FileNotFoundError('No training progress in ' + str(root))
    dest = root / 'PPO_QP_ANALYSIS.zip'
    temporary = dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix.lower() in {'.json', '.jsonl', '.csv', '.log', '.txt'}
                    and not any(part in {'checkpoints', 'checkpoint', 'actions'}
                                for part in path.relative_to(root).parts)):
                z.write(path, path.relative_to(root).as_posix())
    os.replace(temporary, dest)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(export_analysis(args.run_dir))
