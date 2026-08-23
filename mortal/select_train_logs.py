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
    parser = argparse.ArgumentParser(description='filter self-play games where trainee wins')
    parser.add_argument('--src', default='self_play_logs')
    parser.add_argument('--dst', default='train_logs')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for path in sorted(src.glob('*.json.gz')):
        if trainee_score(path) > 0:
            shutil.copy2(path, dst / path.name)
            copied += 1
        else:
            skipped += 1

    print(f'files total: {copied + skipped}, selected: {copied}, skipped: {skipped}')
    print(f'files copied to {dst}: {copied}')


if __name__ == '__main__':
    main()
