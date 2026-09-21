import prelude

import collections
import gc
import gzip
import json
import logging
import os
import random
import time
import torch
from glob import glob
from copy import deepcopy
from datetime import datetime
from itertools import chain
from torch import optim, nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from common import parameter_count
from player import TestPlayer
from dataloader import FileDatasetsIter
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import Brain, CategoricalPolicy
from libriichi.consts import obs_shape
from config import config


def cfg_get(section, key, default=None):
    value = section.get(key)
    return default if value is None else value


def expand_data_files(patterns):
    file_list = []
    for pat in patterns:
        pat = os.path.expanduser(pat)
        if os.path.isdir(pat):
            file_list.extend(glob(os.path.join(pat, '**', '*.json.gz'), recursive=True))
        elif any(ch in pat for ch in '*?['):
            file_list.extend(glob(pat, recursive=True))
        elif os.path.isfile(pat):
            file_list.append(pat)
        else:
            logging.warning(f'no data matched: {pat}')
    file_list = sorted(set(file_list))
    if not file_list:
        raise SystemExit('no log files found, check [train_1v3] globs in config')
    return file_list


def inspect_first_log(file_list):
    for filename in file_list:
        try:
            with gzip.open(filename, 'rt') as f:
                first = json.loads(next(f))
            if first.get('type') == 'start_game':
                return first
        except (OSError, json.JSONDecodeError, StopIteration):
            continue
    return None


def build_optimizer(models, weight_decay, betas, eps):
    decay_params = []
    no_decay_params = []
    for model in models:
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
    return optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)


def train():
    rcfg = config.get('train_1v3')
    if rcfg is None:
        raise SystemExit(
            'missing [train_1v3] section in config (see config.example.toml); '
            'all knobs of this script are configured there, no command line arguments are used'
        )

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

    version = config['control']['version']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    old_update_every = cfg_get(rcfg, 'old_update_every', 0) or config['control']['old_update_every']
    log_every = cfg_get(rcfg, 'log_every', 0) or config['control']['log_every']
    epochs = rcfg['epochs']
    max_steps = rcfg.get('max_steps', 0) or 0
    enable_amp = config['control']['enable_amp']

    data_patterns = rcfg['globs']
    player_names = list(rcfg.get('player_names', [])) or ['trainee']
    file_list = expand_data_files(data_patterns)

    logging.info(f'device: {device}')
    logging.info(f'version: {version}, obs shape: {obs_shape(version)}')
    logging.info(f'data files: {len(file_list):,}')
    logging.info(f'player filter: {player_names}')

    first = inspect_first_log(file_list)
    if first is not None:
        logging.info(f'log meta sample: names={first.get("names")} seed={first.get("seed")}')

    file_data = FileDatasetsIter(
        version=version,
        file_list=file_list,
        player_names=player_names,
        num_epochs=1,
    )
    logging.info('preloading samples...')
    t_preload = datetime.now()
    obs_all, actions_all, masks_all, advantage_all = file_data.preload_awr()
    logging.info(
        f'preloaded {obs_all.shape[0]:,} samples in '
        f'{(datetime.now() - t_preload).total_seconds():.1f}s'
    )
    logging.info(
        f'obs {tuple(obs_all.shape)}, actions {tuple(actions_all.shape)}, '
        f'masks {tuple(masks_all.shape)}, advantage {tuple(advantage_all.shape)}'
    )

    adv_hist = collections.Counter(advantage_all.tolist())
    logging.info(
        f'advantage (= (final match score - 25000) // 1000 of the tracked player, '
        f'same for every decision of a match) mean={advantage_all.mean():.3f} '
        f'std={advantage_all.std():.3f}'
    )
    logging.info(f'advantage histogram: {dict(sorted(adv_hist.items()))}')

    n_total = int(actions_all.shape[0])
    num_batches = n_total // batch_size
    logging.info(f'{n_total:,} samples -> {num_batches:,} batches of {batch_size} (last partial batch dropped)')
    if num_batches == 0:
        raise SystemExit(f'not enough samples ({n_total:,}) for batch_size {batch_size:,}')

    if rcfg.get('inspect', False):
        act_hist = collections.Counter(actions_all.tolist())
        n_neg = int((advantage_all < 0).sum())
        n_zero = int((advantage_all == 0).sum())
        n_pos = int((advantage_all > 0).sum())
        logging.info(f'advantage<0: {n_neg:,}, ==0: {n_zero:,}, >0: {n_pos:,}')
        logging.info(f'action histogram: {dict(sorted(act_hist.items()))}')
        return

    obs_all = torch.as_tensor(obs_all)
    actions_all = torch.as_tensor(actions_all)
    masks_all = torch.as_tensor(masks_all)
    advantage_all = torch.as_tensor(advantage_all)

    loss_mode = rcfg.get('loss', 'online')
    normalize_adv = rcfg.get('normalize_adv', False)
    clip_ratio = config['policy']['clip_ratio']
    dual_clip = config['policy']['dual_clip']
    entropy_weight = config['policy']['entropy_weight']
    awr_beta = config['policy']['awr_beta']
    awr_clip = config['policy']['awr_clip']
    max_grad_norm = config['optim']['max_grad_norm']

    init_state = cfg_get(rcfg, 'init_state', '') or config['control']['state_file']
    out_state = cfg_get(rcfg, 'out_state', '') or (init_state.rsplit('.', 1)[0] + '_1v3train.pth')
    resume = rcfg.get('resume', False)
    save_every = rcfg.get('save_every', 0) or 0
    test_every = rcfg.get('test_every', 0) or 0
    test_games = rcfg.get('test_games', 0) or 0
    seed = rcfg.get('seed', 0) or 0
    tensorboard_dir = rcfg.get('tensorboard_dir', '') or None

    mortal = Brain(version=version, **config['resnet'], Norm='GN').to(device)
    policy_net = CategoricalPolicy().to(device)
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')

    state = None
    if resume and os.path.exists(out_state):
        state = torch.load(out_state, weights_only=False, map_location=device)
        logging.info(f'resuming full trainer state from {out_state} (steps={state.get("steps", "?")})')
    elif os.path.exists(init_state):
        state = torch.load(init_state, weights_only=False, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'initializing networks from {init_state} (saved {timestamp})')
    else:
        logging.warning(f'{init_state} not found, starting from random init')

    if state is not None:
        mortal.load_state_dict(state['mortal'])
        policy_net.load_state_dict(state['policy_net'])
    Old_mortal = deepcopy(mortal)
    Old_policy_net = deepcopy(policy_net)
    if resume and os.path.exists(out_state):
        if 'old_mortal' in state and 'old_policy_net' in state:
            Old_mortal.load_state_dict(state['old_mortal'])
            Old_policy_net.load_state_dict(state['old_policy_net'])

    optimizer = build_optimizer(
        (mortal, policy_net),
        weight_decay=config['optim']['weight_decay'],
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
    )
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    steps = 0
    if resume and os.path.exists(out_state):
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        steps = state.get('steps', 0)

    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None
    test_player = None

    rolling = {'loss': 0.0, 'entropy': 0.0, 'ratio': 0.0, 'clipped': 0.0, 'count': 0}
    epoch_acc = {'loss': 0.0, 'entropy': 0.0, 'ratio': 0.0, 'clipped': 0.0, 'count': 0}
    grad_norm_ema = 0.0

    if device.type == 'cuda':
        logging.info(f'cuda: {torch.cuda.get_device_name(device)}')

    def save_trainer_state(path_, steps_, epoch):
        save_cfg = dict(config)
        save_cfg['control']['device'] = str(device)
        full_state = {
            'mortal': mortal.state_dict(),
            'policy_net': policy_net.state_dict(),
            'old_mortal': Old_mortal.state_dict(),
            'old_policy_net': Old_policy_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps_,
            'epoch': epoch,
            'timestamp': datetime.now().timestamp(),
            'config': save_cfg,
        }
        tmp = path_ + '.tmp'
        torch.save(full_state, tmp)
        os.replace(tmp, path_)
        logging.info(f'state saved to {path_} (steps={steps_:,}, epoch={epoch})')

    def run_test_play():
        nonlocal test_player
        if test_player is None:
            test_player = TestPlayer()
        mortal.eval()
        policy_net.eval()
        games = test_games or config['test_play']['games']
        stat = test_player.test_play(games // 4, mortal, policy_net, device)
        mortal.train()
        policy_net.train()
        avg_pt = stat.avg_pt([90, 45, 0, -135])
        logging.info(f'[test] avg rank: {stat.avg_rank:.4f}, avg pt: {avg_pt:.4f}, '
                     f'1st/2nd/3rd/4th: {stat.rank_1_rate:.3f}/{stat.rank_2_rate:.3f}/'
                     f'{stat.rank_3_rate:.3f}/{stat.rank_4_rate:.3f}, '
                     f'agari: {stat.agari_rate:.3f}, houjuu: {stat.houjuu_rate:.3f}')
        if writer is not None:
            writer.add_scalar('test/avg_rank', stat.avg_rank, steps)
            writer.add_scalar('test/avg_pt', avg_pt, steps)

    def acc(acc_dict, key, value):
        acc_dict[key] += value

    def train_batch(obs, actions, masks, advantage):
        nonlocal steps
        nonlocal grad_norm_ema
        nonlocal Old_mortal
        nonlocal Old_policy_net
        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        advantage = advantage.to(dtype=torch.float32, device=device)
        assert masks[range(advantage.shape[0]), actions].all()

        if normalize_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        if loss_mode == 'online':
            with torch.no_grad():
                with torch.autocast(device.type, enabled=enable_amp):
                    old_dist = Categorical(probs=Old_policy_net(Old_mortal(obs), masks))
                    old_log_prob = old_dist.log_prob(actions)

            with torch.autocast(device.type, enabled=enable_amp):
                dist = Categorical(probs=policy_net(mortal(obs), masks))
                new_log_prob = dist.log_prob(actions)
                ratio = (new_log_prob - old_log_prob).exp()
                loss1 = ratio * advantage
                loss2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                min_loss = torch.min(loss1, loss2)
                clip_loss = torch.where(
                    advantage < 0,
                    torch.max(min_loss, dual_clip * advantage),
                    min_loss,
                )
                entropy = dist.entropy().view(-1, 1)
                entropy_loss = entropy * entropy_weight
                loss = -(clip_loss + entropy_loss).mean()
                clip_frac = (loss1 > min_loss).float().mean()
        else:
            with torch.autocast(device.type, enabled=enable_amp):
                dist = Categorical(probs=policy_net(mortal(obs), masks))
                log_prob = dist.log_prob(actions)
                exp_adv = torch.exp(advantage / awr_beta)
                if awr_clip is not None:
                    exp_adv = torch.clamp(exp_adv, max=awr_clip)
                loss = -(exp_adv * log_prob).mean()
                entropy = dist.entropy().view(-1, 1)
                ratio = torch.ones_like(advantage)

        scaler.scale(loss / opt_step_every).backward()

        with torch.inference_mode():
            m_loss = float(loss.detach().mean().item())
            m_entropy = float(entropy.detach().mean().item())
            m_ratio = float(ratio.detach().mean().item())
            if loss_mode == 'online':
                m_clipped = float(clip_frac.item())
            else:
                m_clipped = 0.0

        acc(rolling, 'loss', m_loss)
        acc(rolling, 'entropy', m_entropy)
        acc(rolling, 'ratio', m_ratio)
        acc(rolling, 'clipped', m_clipped)
        rolling['count'] += 1
        acc(epoch_acc, 'loss', m_loss)
        acc(epoch_acc, 'entropy', m_entropy)
        acc(epoch_acc, 'ratio', m_ratio)
        acc(epoch_acc, 'clipped', m_clipped)
        epoch_acc['count'] += 1

        steps += 1
        if steps % opt_step_every == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                clip_grad_norm_(params, max_grad_norm)
            if enable_amp:
                sq = 0.0
                for p in chain.from_iterable(g['params'] for g in optimizer.param_groups):
                    if p.grad is not None:
                        sq += float(p.grad.float().norm().item() ** 2)
                grad_norm_ema = 0.9 * grad_norm_ema + 0.1 * (sq ** 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if steps % old_update_every == 0 and loss_mode == 'online':
            Old_mortal = deepcopy(mortal)
            Old_policy_net = deepcopy(policy_net)
            logging.info(f'old-policy snapshot refreshed at step {steps}')

        if writer is not None and steps % 10 == 0:
            writer.add_scalar('train/loss', m_loss, steps)
            writer.add_scalar('train/entropy', m_entropy, steps)
            writer.add_scalar('train/ratio', m_ratio, steps)
            writer.add_scalar('train/advantage', float(advantage.mean().item()), steps)
            writer.flush()

        if steps % log_every == 0 and rolling['count'] > 0:
            c = rolling['count']
            msg = (f'step {steps:,}  loss {rolling["loss"] / c:.4f}  '
                   f'entropy {rolling["entropy"] / c:.4f}  ratio {rolling["ratio"] / c:.4f}')
            if loss_mode == 'online':
                msg += f'  clipped {rolling["clipped"] / c:.3f}'
            if grad_norm_ema > 0:
                msg += f'  grad_norm {grad_norm_ema:.3f}'
            logging.info(msg)
            for k in ('loss', 'entropy', 'ratio', 'clipped'):
                rolling[k] = 0.0
            rolling['count'] = 0

    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(int(time.time()) % (2 ** 31))

    mortal.train()
    policy_net.train()

    for epoch in range(1, epochs + 1):
        logging.info(f'=== epoch {epoch}/{epochs} starting (total steps so far: {steps:,}) ===')
        perm = torch.randperm(n_total)
        for k in epoch_acc:
            epoch_acc[k] = 0.0
        for b in range(num_batches):
            if max_steps and steps >= max_steps:
                break
            idxs = perm[b * batch_size:(b + 1) * batch_size]
            train_batch(
                obs_all[idxs],
                actions_all[idxs],
                masks_all[idxs],
                advantage_all[idxs],
            )
            if save_every and steps % save_every == 0:
                save_trainer_state(out_state, steps, epoch)
            if test_every and steps % test_every == 0:
                run_test_play()
        c = epoch_acc['count']
        if c > 0:
            logging.info(
                f'--- epoch {epoch}/{epochs} summary: loss {epoch_acc["loss"] / c:.4f}  '
                f'entropy {epoch_acc["entropy"] / c:.4f}  ratio {epoch_acc["ratio"] / c:.4f}'
                + (f'  clipped {epoch_acc["clipped"] / c:.3f}' if loss_mode == 'online' else '')
            )
        save_trainer_state(out_state, steps, epoch)
        gc.collect()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
