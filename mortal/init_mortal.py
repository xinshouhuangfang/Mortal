#!/usr/bin/env python3
"""Randomly initialize a fresh mortal.pth checkpoint (steps=0, random weights).

The saved state follows the format written by train_offline_phase1.py, so it can
be loaded by both train_offline_phase1.py and play_human.py. The model
architecture (version, conv_channels, num_blocks) and the optimizer/scheduler
come from the active config.toml.

Usage:
  python init_mortal.py
  python init_mortal.py --output my_checkpoint.pth
"""
import prelude

import argparse
import torch
from datetime import datetime
from itertools import chain
from torch import optim, nn
from torch.amp import GradScaler
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import Brain, CategoricalPolicy
from config import config

def main():
    parser = argparse.ArgumentParser(description='随机初始化 mortal.pth')
    parser.add_argument(
        '--output',
        default=config['control']['state_file'],
        help='输出文件路径(默认取 config 的 state_file)',
    )
    args = parser.parse_args()

    device = torch.device(config['control']['device'])
    version = config['control']['version']
    enable_amp = config['control']['enable_amp']
    betas = config['optim']['betas']
    eps = config['optim']['eps']
    weight_decay = config['optim']['weight_decay']

    mortal = Brain(version=version, **config['resnet'], Norm="GN")
    policy_net = CategoricalPolicy()

    decay_params = []
    no_decay_params = []
    for model in (mortal, policy_net):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    state = {
        'mortal': mortal.state_dict(),
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': 0,
        'timestamp': datetime.now().timestamp(),
        'best_perf': {'avg_rank': 4., 'avg_pt': -135.},
        'config': config,
        'shared_stats': {'count': 0, 'mean': 0, 'variance': 0},
    }
    torch.save(state, args.output)
    print(f'saved {args.output} (steps=0)')

if __name__ == '__main__':
    main()
