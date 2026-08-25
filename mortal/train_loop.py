import prelude

import os
import shutil
import logging
import subprocess
import sys
from datetime import datetime
from os import path
from config import config

def script_dir():
    return path.dirname(path.abspath(__file__))

def run(cmd):
    logging.info(f'$ {" ".join(cmd)}')
    subprocess.run(cmd, cwd = script_dir(), check = True)

def main():
    loop = config['loop']
    iterations = loop.get('iterations', 0)
    selfplay_target = loop.get('selfplay_target', 2000)

    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    baseline_file = config['baseline']['test']['state_file']
    train_logs_dir = path.join(script_dir(), 'train_logs')
    file_index = config['dataset']['file_index']
    if not path.isabs(file_index):
        file_index = path.join(script_dir(), file_index)

    if not path.exists(best_state_file):
        assert path.exists(baseline_file), f'{baseline_file} not found'
        shutil.copy(baseline_file, best_state_file)
        logging.info(f'initialized {best_state_file} from {baseline_file}')

    iteration = 0
    while iterations == 0 or iteration < iterations:
        iteration += 1
        logging.info(f'--- iteration {iteration} ---')

        # 1. start training from the best model
        shutil.copy(best_state_file, state_file)
        logging.info(f'copied {best_state_file} -> {state_file}')

        # 2. self-play and collect >= selfplay_target winning samples
        run([sys.executable, path.join(script_dir(), 'run_selfplay_loop.py'), str(selfplay_target)])

        # 3. drop the cached file index so train.py rebuilds it from the fresh train_logs
        if path.exists(file_index):
            os.remove(file_index)
            logging.info(f'removed stale file index {file_index}')

        # 4. train; train.py may promote mortal to best via test_every
        run([sys.executable, path.join(script_dir(), 'train.py')])

        # 5. move train_logs away so the next iteration collects fresh samples
        if path.isdir(train_logs_dir):
            backup = f'{train_logs_dir}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            shutil.move(train_logs_dir, backup)
            logging.info(f'renamed {train_logs_dir} -> {backup}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
