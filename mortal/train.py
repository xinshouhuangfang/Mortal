def train():
    import prelude

    import logging
    import gc
    import gzip
    import json
    import queue
    import random
    import threading
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.tensorboard import SummaryWriter
    from common import parameter_count, filtered_trimmed_lines, tqdm
    from dataloader import FileDatasetsIter
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    log_every = config['control']['log_every']
    min_q_weight = config['cql']['min_q_weight']

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    all_models = (mortal, dqn)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
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

    steps = 0
    state_file = config['control']['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
    }
    all_q = torch.zeros((log_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((log_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def preload_dataset():
        player_names_set = set()
        for filename in config['dataset']['player_names_files']:
            with open(filename) as f:
                player_names_set.update(filtered_trimmed_lines(f))
        player_names = list(player_names_set)
        logging.info(f'loaded {len(player_names):,} players')

        file_index = config['dataset']['file_index']
        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            file_list = index['file_list']
        else:
            logging.info('building file index...')
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            if len(player_names_set) > 0:
                filtered = []
                for filename in tqdm(file_list, unit='file'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )

        # Preload the whole dataset into memory once, then let each train_epoch
        # feed the GPU from the same tensors.
        t_preload = datetime.now()
        try:
            obs_all, actions_all, masks_all, steps_all, rewards_all = file_data.preload()
        except ValueError as ex:
            logging.warning(f'training skipped: {ex}')
            return None
        n_total = actions_all.shape[0]
        logging.info(
            f'preloaded {n_total:,} samples in {(datetime.now() - t_preload).total_seconds():.1f}s'
        )

        return (
            torch.as_tensor(obs_all),
            torch.as_tensor(actions_all),
            torch.as_tensor(masks_all),
            torch.as_tensor(steps_all),
            torch.as_tensor(rewards_all),
            n_total,
        )

    dataset = preload_dataset()
    if dataset is None:
        return
    obs_all, actions_all, masks_all, steps_all, rewards_all, n_total = dataset
    num_batches = n_total // batch_size

    def train_epoch():
        nonlocal steps
        nonlocal idx

        pb = tqdm(total=None, desc='TRAIN')
        log_dqn_loss = 0
        log_cql_loss = 0

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards):
            nonlocal steps
            nonlocal idx
            nonlocal log_dqn_loss
            nonlocal log_cql_loss

            assert masks[range(batch_size), actions].all()

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                stats['cql_loss'] += cql_loss
                log_dqn_loss += dqn_loss.item()
                log_cql_loss += cql_loss.item()
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if steps % log_every == 0:
                logging.info(
                    f'step: {steps:,} | '
                    f'dqn_loss: {log_dqn_loss / log_every:.6f} | '
                    f'cql_loss: {log_cql_loss / log_every:.6f} | '
                    f'lr: {scheduler.get_last_lr()[0]:.3e}'
                )
                log_dqn_loss = 0
                log_cql_loss = 0

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / log_every, steps)
                writer.add_scalar('loss/cql_loss', stats['cql_loss'] / log_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

        perm = torch.randperm(n_total)

        # use a dedicated stream for H2D copies so they overlap with the
        # forward/backward running on the default stream
        copy_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        batch_queue = queue.Queue(maxsize=2)
        producer_error = []

        def producer():
            try:
                for b in range(num_batches):
                    idxs = perm[b * batch_size:(b + 1) * batch_size]
                    if copy_stream is not None:
                        with torch.cuda.stream(copy_stream):
                            batch = (
                                obs_all[idxs].to(device, non_blocking=True),
                                actions_all[idxs].to(device, non_blocking=True),
                                masks_all[idxs].to(device, non_blocking=True),
                                steps_all[idxs].to(device, non_blocking=True),
                                rewards_all[idxs].to(dtype=torch.float64, device=device, non_blocking=True),
                            )
                        copy_stream.synchronize()
                    else:
                        batch = (
                            obs_all[idxs].to(device),
                            actions_all[idxs].to(device),
                            masks_all[idxs].to(device),
                            steps_all[idxs].to(device),
                            rewards_all[idxs].to(dtype=torch.float64, device=device),
                        )
                    batch_queue.put(batch)
                batch_queue.put(None)
            except Exception as ex:
                producer_error.append(ex)
                batch_queue.put(None)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        for _ in range(num_batches):
            batch = batch_queue.get()
            if batch is None:
                break
            train_batch(*batch)

        producer_thread.join()
        pb.close()
        if producer_error:
            raise producer_error[0]

        state = {
            'mortal': mortal.state_dict(),
            'current_dqn': dqn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'config': config,
        }
        torch.save(state, state_file)
        logging.info(f'training done, saved at step {steps:,}')

    # train multiple epochs on the preloaded dataset (loaded once above)
    for i in range(1000000):
        train_epoch()
        gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def main():
    train()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
