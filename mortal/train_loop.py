import prelude

import os
import sys
import shutil
import logging
import secrets
import subprocess
import numpy as np
import torch
from os import path
from glob import glob
from model import Brain, DQN
from engine import MortalEngine
from player import TestPlayer
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
    eval_threshold = loop.get('eval_threshold', 0.0)
    eval_games = loop.get('eval_games', 64)

    device = torch.device(config['control']['device'])
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    log_dir = config['1v3']['log_dir']
    file_index = config['dataset']['file_index']

    if loop.get('start_from_baseline', False):
        assert path.exists(best_state_file), f'{best_state_file} not found'
        shutil.copy(best_state_file, state_file)
        logging.info(f'initialized {state_file} from {best_state_file}')
        set_start_from_baseline(False)

    ex = loop['exploration']
    epsilon_initial = ex.get('epsilon_initial', 0.1)
    epsilon_final = ex.get('epsilon_final', 0.005)
    temp_initial = ex.get('temp_initial', 0.5)
    temp_final = ex.get('temp_final', 0.05)
    decay_iters = ex.get('decay_iters', 20)

    challenger_cfg = config['1v3']['challenger']
    champion_cfg = config['1v3']['champion']
    seeds_per_iter = config['1v3']['games_per_iter'] // 4
    seed_key = config['1v3'].get('seed_key', -1)
    if seed_key == -1:
        seed_key = secrets.randbits(16)

    test_player = TestPlayer()
    script_dir = path.dirname(path.abspath(__file__))

    iteration = 0
    while iterations == 0 or iteration < iterations:
        iteration += 1
        t = min((iteration - 1) / max(decay_iters, 1), 1.0)
        epsilon = epsilon_initial + (epsilon_final - epsilon_initial) * t
        temp = temp_initial + (temp_final - temp_initial) * t
        logging.info(f'--- iteration {iteration} | epsilon={epsilon:.4f} temp={temp:.4f} ---')

        # 1. generate fresh self-play data (trainee vs frozen baseline)
        if path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        challenger = load_engine(challenger_cfg, device, boltzmann_epsilon=epsilon, boltzmann_temp=temp)
        champion = load_engine(champion_cfg, device)
        env = OneVsThree(disable_progress_bar=False, log_dir=log_dir)
        rankings = env.py_vs_py(
            challenger = challenger,
            champion = champion,
            seed_start = (200 + (iteration - 1) * seeds_per_iter, seed_key),
            seed_count = seeds_per_iter,
        )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        logging.info(f'gen: challenger rankings {rankings} ({avg_rank:.4f} rank, {avg_pt:.4f}pt)')

        generated = glob(path.join(log_dir, '*.json.gz'))
        assert len(generated) > 0, 'no games generated'

        # 2. force rebuild of the file index against the fresh logs
        if path.exists(file_index):
            os.remove(file_index)

        # 3. train one epoch
        subprocess.run(
            [sys.executable, path.join(script_dir, 'train.py')],
            cwd = script_dir,
            check = True,
        )

        # 4. evaluate challenger vs baseline
        state = torch.load(state_file, weights_only=True, map_location=device)
        c = state['config']
        version = c['control'].get('version', 1)
        conv_channels = c['resnet']['conv_channels']
        num_blocks = c['resnet']['num_blocks']
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).to(device).eval()
        dqn = DQN(version=version).to(device).eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])

        stat = test_player.test_play(eval_games // 4, mortal, dqn, device)
        avg_pt = stat.avg_pt([90, 45, 0, -135])
        logging.info(f'eval: avg_rank={stat.avg_rank:.4f} avg_pt={avg_pt:.4f}')

        # 5. promote
        if avg_pt >= eval_threshold:
            shutil.copy(state_file, best_state_file)
            logging.info(f'promoted: {state_file} -> {best_state_file} (avg_pt={avg_pt:.4f})')
        else:
            logging.info(f'no promotion: avg_pt={avg_pt:.4f} < {eval_threshold}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
