import torch
from torch import optim, nn
from torch.amp import GradScaler
from datetime import datetime

from config import config
from model import Brain, DQN
from lr_scheduler import LinearWarmUpCosineAnnealingLR


def main():
    version = config['control']['version']
    device = torch.device(config['control']['device'])
    enable_amp = config['control']['enable_amp']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    all_models = (mortal, dqn)

    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']

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

    state = {
        'mortal': mortal.state_dict(),
        'current_dqn': dqn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': 0,
        'timestamp': datetime.now().timestamp(),
        'config': config,
    }

    state_file = config['control']['state_file']
    torch.save(state, state_file)
    print(f'saved randomly-initialized checkpoint to {state_file}')


if __name__ == '__main__':
    main()
