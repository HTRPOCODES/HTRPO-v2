from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss

DDPGconfig = {
    'steps_per_iter': 50,
    'learn_start_step': 10000,
    'memory_size': 1e6,
    'batch_size': 128,
    'reward_decay': 0.99,
    'noise_var' : 0.3,
    'noise_min' : 0.01,
    'noise_decrease' : 2e-5,
    'optimizer': optim.Adam,
    'loss': MSELoss,
    'action_bounds':1,
    'max_grad_norm': None,
    'tau' : 5e-3,
    'lr': 1e-4,
    'v_optimizer': optim.Adam,
    'lr_v' : 1e-3,
    'hidden_layers': [400, 300],
    'hidden_layers_v' : [400, 300],
    'loss_func_v': MSELoss,
    'act_func': F.relu,
    'out_act_func': F.tanh,
    'using_bn': False,
}
