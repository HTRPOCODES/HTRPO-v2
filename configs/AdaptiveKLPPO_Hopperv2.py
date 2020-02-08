from torch import optim
import torch.nn.functional as F

AdaptiveKLPPOconfig = {
        'action_bounds': 1,
        'reward_decay': 0.99,
        'steps_per_iter': 2048,
        'nbatch_per_iter': 32,
        'updates_per_iter': 10,
        'max_grad_norm': 0.25,
        'GAE_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 3e-4,
        'lr_v': 3e-4,
        'hidden_layers' : [64,64],
        'hidden_layers_v' : [64,64],
        'entropy_weight' : 0,
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC',
        'act_func':F.tanh,
        'out_act_func': F.tanh,
    }

AdaptiveKLPPOconfig['memory_size'] = AdaptiveKLPPOconfig['steps_per_iter']