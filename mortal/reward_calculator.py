import torch
import numpy as np
from multiprocessing import Manager, Value

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False, shared_stats=None):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval() if grp is not None else None
        self.uniform_init = uniform_init

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

        if shared_stats is None:
            manager = Manager()
            self.shared_stats = {
                'count': manager.Value('i', 0),
                'mean': manager.Value('d', 0.0),
                'M2': manager.Value('d', 0.0),
                'lock': manager.Lock()
            }
        else:
            self.shared_stats = shared_stats

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix
    
    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob
    
    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        reward_np = reward.cpu().numpy()

        if len(reward_np) == 0:
            return reward_np

        with self.shared_stats['lock']:
            global_count = self.shared_stats['count'].value
            global_mean = self.shared_stats['mean'].value
            global_M2 = self.shared_stats['M2'].value

            # per-sample online Welford update (batch = rewards of one game,
            # which may be a single value for single-kyoku logs; pooling
            # variance over all individual rewards keeps the running std from
            # collapsing to ~0 in that case)
            for r in reward_np:
                global_count += 1
                delta = r - global_mean
                global_mean += delta / global_count
                delta2 = r - global_mean
                global_M2 += delta * delta2

            std = 1.0 if global_count < 2 else max(np.sqrt(global_M2 / global_count), 1e-8)

            self.shared_stats['count'].value = global_count
            self.shared_stats['mean'].value = global_mean
            self.shared_stats['M2'].value = global_M2

        standardized = (reward_np - global_mean) / std
        return standardized
