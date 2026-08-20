import argparse
import gzip
import json
import shutil
from pathlib import Path

SPLIT_SEATS = {'a': 0, 'b': 1, 'c': 2, 'd': 3}


def trainee_score(path):
    seat = SPLIT_SEATS[path.stem.rsplit('.', 1)[0].split('_')[-1]]
    score = 0
    with gzip.open(path, 'rt') as f:
        for line in f:
            ev = json.loads(line)
            if 'deltas' in ev:
                score += ev['deltas'][seat]
    return score


def main():
    parser = argparse.ArgumentParser(description='filter self-play games by trainee total score')
    parser.add_argument('--src', default='self_play_logs')
    parser.add_argument('--dst', default='train_logs')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    groups = {}
    for path in sorted(src.glob('*.json.gz')):
        parts = path.stem.rsplit('.', 1)[0].split('_')
        seed = (parts[0], parts[1])
        groups.setdefault(seed, []).append(path)

    selected = 0
    skipped = 0
    copied = 0
    for seed, paths in sorted(groups.items()):
        total = sum(trainee_score(p) for p in paths)
        if total != 0:
            for p in paths:
                shutil.copy2(p, dst / p.name)
                copied += 1
            selected += 1
        else:
            skipped += 1

    print(f'groups total: {len(groups)}, selected: {selected}, skipped: {skipped}')
    print(f'files copied to {dst}: {copied}')


if __name__ == '__main__':
    main()
