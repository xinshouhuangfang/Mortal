import prelude

import numpy as np
import argparse
import torch
import secrets
import os
from model import Brain,CategoricalPolicy
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

def main():
    parser = argparse.ArgumentParser(description='挑战者 vs 冠军(1v3)')
    parser.add_argument('games', nargs='?', type=int, default=None,
                        help='对战局数(不传则用 config 中 1v3.games_per_iter 配置)')
    args = parser.parse_args()

    cfg = config['1v3']
    games_per_iter = args.games if args.games is not None else cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = false

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(16)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cham_cfg = state['config']
        version = cham_cfg['control'].get('version', 1)
        conv_channels = cham_cfg['resnet']['conv_channels']
        num_blocks = cham_cfg['resnet']['num_blocks']
        mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,Norm = "GN").eval()
        dqn = CategoricalPolicy().eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['policy_net'])
        if cfg['champion']['enable_compile']:
            mortal.compile()
            dqn.compile()
        engine_cham = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = version,
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            name = cfg['champion']['name'],
        )

    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    chal_cfg = state['config']
    version = chal_cfg['control'].get('version', 1)
    conv_channels = chal_cfg['resnet']['conv_channels']
    num_blocks = chal_cfg['resnet']['num_blocks']
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,Norm = "GN").eval()
    dqn = CategoricalPolicy().eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['policy_net'])
    if cfg['challenger']['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine_chal = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = torch.device(cfg['challenger']['device']),
        enable_amp = cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
        name = cfg['challenger']['name'],
    )

    seed_start = 200
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        total_score = env.py_vs_py(
            challenger = engine_chal,
            champion = engine_cham,
            seed_start = (seed, key),
            seed_count = seeds_per_iter,
        )
        games = seeds_per_iter * 4
        avg_score = total_score / games
        print(f'challenger total_score: {total_score} ({avg_score} /game)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
