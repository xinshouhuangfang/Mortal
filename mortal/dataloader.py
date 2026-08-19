import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def preload(self):
        # Encode the whole dataset once into memory as dense arrays, then let
        # training feed the GPU by slicing them directly.
        obs_l = []
        actions_l = []
        masks_l = []
        steps_l = []
        rewards_l = []
        total = 0

        for _ in range(self.num_epochs):
            for augmented in self._augment_order():
                random.shuffle(self.file_list)
                self.loader = GameplayLoader(
                    version = self.version,
                    oracle = self.oracle,
                    player_names = self.player_names,
                    excludes = self.excludes,
                    augmented = augmented,
                )
                data = self.loader.load_gz_log_files(self.file_list)
                for file in data:
                    for game in file:
                        obs_list = game.take_obs()
                        game_size = len(obs_list)
                        if game_size == 0:
                            continue
                        obs = np.stack(obs_list)
                        actions = np.asarray(game.take_actions())
                        masks = np.stack(game.take_masks())
                        _ = game.take_at_kyoku()
                        dones = np.asarray(game.take_dones())
                        apply_gamma = np.asarray(game.take_apply_gamma())

                        player_id = game.take_player_id()
                        final_scores = np.asarray(game.take_grp().take_final_scores())

                        kyoku_rewards = (final_scores[player_id] - 25000) // 1000

                        steps_to_done = np.zeros(game_size, dtype=np.int64)
                        for i in reversed(range(game_size)):
                            if not dones[i]:
                                steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                        obs_l.append(obs)
                        actions_l.append(actions)
                        masks_l.append(masks)
                        steps_l.append(steps_to_done)
                        rewards_l.append(np.full(game_size, kyoku_rewards, dtype=np.int64))
                        total += game_size

        if total == 0:
            raise ValueError('no training samples found')

        return (
            np.concatenate(obs_l),
            np.concatenate(actions_l),
            np.concatenate(masks_l),
            np.concatenate(steps_l),
            np.concatenate(rewards_l),
        )

    def _augment_order(self):
        if not self.enable_augmentation:
            return (self.augmented_first,)
        return (self.augmented_first, not self.augmented_first)

    def __iter__(self):
        raise NotImplementedError('use preload()')

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
