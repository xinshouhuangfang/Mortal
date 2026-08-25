import codecs
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
LOG_FILE = BASE / 'log.txt'
STATE_FILE = BASE / 'mortal.pth'
BEST_FILE = BASE / 'best.pth'

PIPELINE = [
    'python self_play.py',
    'rm train_logs/*',
    'cp self_play_logs/* train_logs/',
    'rm file_index.pth',
    'python train.py',
]


def read_best_score():
    if not BEST_FILE.exists():
        return 0.0
    state = torch.load(BEST_FILE, weights_only=True, map_location='cpu')
    return state.get('test_score', 0.0)


def run(cmd, log_file):
    print(f'$ {cmd}', flush=True)
    log_file.write(f'$ {cmd}\n')
    log_file.flush()
    proc = subprocess.Popen(
        cmd, shell=True, cwd=BASE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    )
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    output = []
    while True:
        chunk = os.read(proc.stdout.fileno(), 4096)
        if not chunk:
            break
        text = decoder.decode(chunk)
        sys.stdout.write(text)
        sys.stdout.flush()
        log_file.write(text)
        log_file.flush()
        output.append(text)
    decoder.decode(b'', final=True)
    returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(f'command failed (rc={returncode}): {cmd}')
    return ''.join(output)


def main():
    best = read_best_score()
    print(f'start: current best score = {best} (from {BEST_FILE})', flush=True)

    with open(LOG_FILE, 'a', buffering=1) as log_file:
        i = 0
        while True:
            i += 1
            header = f'=== iteration {i} @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, current best = {best} ==='
            print(header, flush=True)
            log_file.write(header + '\n')
            log_file.flush()

            for cmd in PIPELINE:
                run(cmd, log_file)

            score = parse_total_score(run('python one_vs_three.py', log_file))
            print(f'one_vs_three total_score: {score}', flush=True)
            log_file.write(f'one_vs_three total_score: {score}\n')
            log_file.flush()

            if score > best:
                snapshot = BASE / f'best-{score}.pth'
                shutil.copy(STATE_FILE, snapshot)
                state = torch.load(STATE_FILE, weights_only=True, map_location='cpu')
                state['test_score'] = score
                torch.save(state, BEST_FILE)
                msg = f'>>> new best! {score} > {best}; saved {snapshot.name}, updated {BEST_FILE}'
                print(msg, flush=True)
                log_file.write(msg + '\n')
                log_file.flush()
                best = score
            else:
                msg = f'>>> no improvement: {score} <= {best}'
                print(msg, flush=True)
                log_file.write(msg + '\n')
                log_file.flush()


def parse_total_score(output):
    import re
    matches = re.findall(r'challenger total_score: (-?\d+)', output)
    if not matches:
        raise SystemExit(f'could not parse total_score from one_vs_three output:\n{output}')
    return int(matches[-1])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('interrupted by user', flush=True)
