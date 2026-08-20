import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def count_files():
    return sum(1 for _ in BASE.glob('train_logs/*.json.gz'))


def run(cmd):
    print(f'$ {cmd}', flush=True)
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        raise SystemExit(f'command failed: {cmd}')


def main():
    parser = argparse.ArgumentParser(
        description='repeatedly self-play and select logs until train_logs is full'
    )
    parser.add_argument('target', nargs='?', type=int, default=10000)
    args = parser.parse_args()

    count = count_files()
    print(f'start: train_logs={count}, target={args.target}', flush=True)

    i = 0
    while count < args.target:
        i += 1
        print(f'=== iteration {i}, train_logs={count}, target={args.target} ===', flush=True)
        run('python self_play.py')
        run('python select_train_logs.py')
        count = count_files()

    print(f'=== done, train_logs={count} ===', flush=True)


if __name__ == '__main__':
    main()
