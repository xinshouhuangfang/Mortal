#!/usr/bin/env python3
"""Play a mahjong game as a human against the Mortal AI.

The human sits at East (seat 0) and plays 1 hanchan against 3 Mortal AI.
Game logs are saved to `log_dir` (default play_human_logs/) as .json.gz.

Usage:
  python play_human.py
  python play_human.py --state mortal.pth --log_dir play_human_logs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import secrets
import torch
from config import config
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree

def build_engine(state_file, device):
    state = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
    c = state['config']
    version = c['control'].get('version', 1)
    mortal = Brain(
        version=version,
        conv_channels=c['resnet']['conv_channels'],
        num_blocks=c['resnet']['num_blocks'],
    ).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='mortal',
    )

def main():
    parser = argparse.ArgumentParser(description='人类 vs Mortal AI(你坐东家)')
    parser.add_argument('--state', default='mortal.pth', help='模型权重文件(默认 mortal.pth)')
    parser.add_argument('--log_dir', default='play_human_logs', help='牌谱保存目录')
    parser.add_argument('--gui', action='store_true', help='使用图形界面(tkinter)')
    args = parser.parse_args()

    device = torch.device(config['control']['device'])
    engine = build_engine(args.state, device)

    if args.gui:
        from human_gui import run_gui
        seed_key = secrets.randbits(16)
        scores = run_gui(
            engine=engine,
            log_dir=args.log_dir,
            seed_start=(60000, seed_key),
            state_file=args.state,
        )
        if scores is None:
            print('已中止(窗口关闭时游戏未结束)')
            return
    else:
        print('你将作为东家(座位0)对战 3 个 Mortal AI,共 1 局。')
        print('每次轮到你时,按提示输入:牌名(如 1m, 5p, E)或数字,')
        print('或指令 riichi / tsumo / ron / chi_l / chi_m / chi_h / pon /')
        print('daiminkan / kakan / ankan / ryukyoku / pass / none。')
        print()

        env = OneVsThree(disable_progress_bar=True, log_dir=args.log_dir)
        seed_key = secrets.randbits(16)
        scores = env.human_vs_py(
            engine=engine,
            seed_start=(60000, seed_key),
            seed_count=1,
        )

    print()
    print('===== 结果(每局净分差 = 终局分 - 25000) =====')
    total = 0
    for i, s in enumerate(scores, 1):
        total += s[0]
        print(f'半庄{i}: 你(东家) {s[0]:+6d} | Mortal {s[1]:+6d} / {s[2]:+6d} / {s[3]:+6d}')
    print(f'你的总净分: {total:+d}')

if __name__ == '__main__':
    main()
