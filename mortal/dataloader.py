import os
import atexit
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

        # dump each fed sample to a file for debugging; path from
        # env MORTAL_SAMPLE_DUMP, defaults to ./sample_dump.txt
        dump_path = os.environ.get('MORTAL_SAMPLE_DUMP', 'sample_dump.txt')
        self._dump_f = open(dump_path, 'w', encoding='utf-8')
        atexit.register(self._dump_f.close)

    def _dump_sample(self, tag, game_idx, i, obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
        ob = obs[i]
        ob_flat = ob.ravel()
        nonzero = np.flatnonzero(ob_flat)
        samples = ob_flat[nonzero[:10]]
        self._dump_f.write(
            'file={tag} game={game_idx} step={i} obs_shape={shape} dtype={dtype} '
            'obs_nz_count={nz} obs_sample={samples} '
            'action={a} legal={legal} steps_to_done={sd} kyoku_reward={kr} player_rank={pr}\n'.format(
                tag=tag, game_idx=game_idx, i=i, shape=ob.shape, dtype=ob.dtype,
                nz=len(nonzero), samples=samples.tolist() if len(samples) else [],
                a=actions[i], legal=masks[i].tolist(), sd=steps_to_done[i],
                kr=kyoku_rewards, pr=player_ranks,
            )
        )
        self._dump_f.flush()

    def build_iter(self):
        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        tag = os.path.basename(file_list[0]) if file_list else 'unknown'
        game_idx = 0
        for file in data:
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                _ = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # per game
                _ = game.take_player_id()

                game_size = len(obs)

                kyoku_rewards = 6.0
                player_ranks = 1

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    self._dump_sample(
                        tag, game_idx, i, obs, actions, masks,
                        steps_to_done, kyoku_rewards, player_ranks,
                    )
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards,
                        player_ranks,
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)
                game_idx += 1

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
