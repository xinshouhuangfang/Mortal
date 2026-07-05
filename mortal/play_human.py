#!/usr/bin/env python3
"""Play a mahjong game as a human against a tsumogiri AI.

Usage:
  python play_human.py          # play 1 game (4 hanchans)
  python play_human.py --games 2  # play 2 games
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mortal'))

from libriichi.arena import OneVsThree

class TsumogiriEngine:
    def __init__(self, name="tsumogiri"):
        self.engine_type = 'mortal'
        self.name = name
        self.is_oracle = False
        self.version = 4
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = False
        self.device = 'cpu'

    def react_batch(self, obs, masks, invisible_obs):
        import numpy as np
        actions = []
        for i in range(len(obs)):
            mask = np.array(masks[i])
            valid = [j for j in range(37) if mask[j]]
            action = valid[0] if valid else 45
            actions.append(action)
        return actions, [[0.0]*46]*len(obs), [list(m) for m in masks], [True]*len(obs)
    def start_game(self, game_idx): pass
    def end_kyoku(self, game_idx): pass
    def end_game(self, game_idx, scores): pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1, help='number of games (1 game = 4 hanchans)')
    args = parser.parse_args()

    env = OneVsThree(disable_progress_bar=False)
    champ = TsumogiriEngine(name="tsumogiri")

    print(f"Playing {args.games} game(s) as human vs tsumogiri...")
    print("For each action, type the tile name (e.g. 1m, 5p, E),")
    print("a number (0-36), or a command (riichi, pon, chi_l, etc.)")
    print("Type 'pass' or 'none' to skip an action.\n")

    rankings = env.human_vs_py(
        engine=champ,
        seed_start=(50000, 0x1234),
        seed_count=args.games,
    )
    print(f"\nFinal rankings: {rankings}")

if __name__ == '__main__':
    main()
