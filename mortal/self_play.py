import prelude

import os
import shutil
import torch
import secrets
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

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
    cfg = config['self_play']
    games = cfg.get('games', 1024)
    log_dir = cfg.get('log_dir', 'self_play_logs')
    epsilon = cfg.get('boltzmann_epsilon', 0)
    temp = cfg.get('boltzmann_temp', 0.5)
    top_p = cfg.get('top_p', 1.0)
    seed_key = cfg.get('seed_key', -1)
    if seed_key == -1:
        seed_key = secrets.randbits(16)

    challenger_cfg = config['1v3']['challenger']
    champion_cfg = config['1v3']['champion']

    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    challenger = load_engine(challenger_cfg, torch.device(challenger_cfg['device']),
                             boltzmann_epsilon=epsilon, boltzmann_temp=temp, top_p=top_p)
    champion = load_engine(champion_cfg, torch.device(champion_cfg['device']))

    seeds = games // 4
    env = OneVsThree(disable_progress_bar = False, log_dir = log_dir)
    total_score = env.py_vs_py(
        challenger = challenger,
        champion = champion,
        seed_start = (200, seed_key),
        seed_count = seeds,
    )
    n_games = seeds * 4
    avg_score = total_score / n_games
    print(f'challenger total_score: {total_score} ({avg_score:.3f} /game) over {n_games} games')
    print(f'logs saved to {log_dir}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
