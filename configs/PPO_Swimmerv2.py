from torch import optim
import torch.nn.functional as F

PPOconfig = {
        'action_bounds': 1,
        'reward_decay': 0.99,
        'steps_per_iter': 2048,
        'nbatch_per_iter': 32,
        'updates_per_iter': 10,
        'max_grad_norm': 0.5,
        'GAE_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 3e-4,
        'v_coef': 0.5,
        'hidden_layers' : [64,64],
        'hidden_layers_v' : [64,64],
        'entropy_weight' : 0,
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC',
        'act_func':F.tanh,
    }

PPOconfig['memory_size'] = PPOconfig['steps_per_iter']
PPOconfig['lr_v'] = PPOconfig['lr']
