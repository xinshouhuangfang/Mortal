import os
import toml

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

def _resolve_device(value):
    if value != 'auto':
        return value
    override = os.environ.get('MORTAL_DEVICE')
    if override:
        return override
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def _resolve(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'device':
                obj[k] = _resolve_device(v)
            else:
                _resolve(v)
    elif isinstance(obj, list):
        for item in obj:
            _resolve(item)

_resolve(config)
