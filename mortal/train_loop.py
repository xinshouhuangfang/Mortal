import prelude

import os
import sys
import shutil
import logging
import secrets
import subprocess
import torch
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

def config_path():
    return os.environ.get('MORTAL_CFG', 'config.toml')

def set_start_from_baseline(value: bool):
    fp = config_path()
    with open(fp, encoding='utf-8') as f:
        text = f.read()
    text = text.replace('start_from_baseline = true', 'start_from_baseline = false')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(text)

def load_engine(cfg, device, *, boltzmann_epsilon=0, boltzmann_temp=1, top_p=1):
    state = torch.load(cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
    c = state['config']
    version = c['control'].get('version', 1)
    conv_channels = c['resnet']['conv_channels']
    num_blocks = c['resnet']['num_blocks']
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    if cfg['enable_compile']:
        mortal.compile()
        dqn.compile()
    return MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = device,
        stochastic_latent = cfg.get('stochastic_latent', False),
        enable_amp = cfg.get('enable_amp', True),
        enable_rule_based_agari_guard = cfg.get('enable_rule_based_agari_guard', False),
        name = cfg['name'],
        boltzmann_epsilon = boltzmann_epsilon,
        boltzmann_temp = boltzmann_temp,
        top_p = top_p,
    )

def main():
    loop = config['loop']
    iterations = loop.get('iterations', 0)
    challenge_games = loop.get('challenge_games', 128)
    challenge_threshold = loop.get('challenge_threshold', 0.1)

    device = torch.device(config['control']['device'])
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']

    if loop.get('start_from_baseline', False):
        assert path.exists(best_state_file), f'{best_state_file} not found'
        shutil.copy(best_state_file, state_file)
        logging.info(f'initialized {state_file} from {best_state_file}')
        set_start_from_baseline(False)

    challenger_cfg = config['1v3']['challenger']
    champion_cfg = config['1v3']['champion']
    seed_key = config['1v3'].get('seed_key', -1)
    if seed_key == -1:
        seed_key = secrets.randbits(16)

    script_dir = path.dirname(path.abspath(__file__))

    iteration = 0
    while iterations == 0 or iteration < iterations:
        iteration += 1
        logging.info(f'--- iteration {iteration} ---')

        # 1. train one round
        subprocess.run(
            [sys.executable, path.join(script_dir, 'train.py')],
            cwd = script_dir,
            check = True,
        )

        # 2. mortal as challenger vs 3 baseline models
        challenger = load_engine(challenger_cfg, device)
        champion = load_engine(champion_cfg, device)
        env = OneVsThree(disable_progress_bar = False)
        seeds = challenge_games // 4
        total_score = env.py_vs_py(
            challenger = challenger,
            champion = champion,
            seed_start = (10000 + (iteration - 1) * seeds, seed_key),
            seed_count = seeds,
        )
        games = seeds * 4
        avg_score = total_score / games
        logging.info(f'challenge: total_score={total_score} avg_score={avg_score:.4f} (threshold={challenge_threshold})')

        # 3. stop training once the challenge succeeds
        if avg_score > challenge_threshold:
            logging.info(f'challenge succeeded: avg_score={avg_score:.4f} > {challenge_threshold}, stop training')
            return
        logging.info(f'challenge failed: avg_score={avg_score:.4f} <= {challenge_threshold}, train again')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
