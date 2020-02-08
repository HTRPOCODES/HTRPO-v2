from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss

DDPGconfig = {
    'memory_size': 2000,
    'batch_size': 32,
    'reward_decay': 0.99,
    'noise_var' : 1.,
    'noise_min' : 0.01,
    'noise_decrease' : 1e-6,
    'optimizer': optim.Adam,
    'loss': MSELoss,
    'action_bounds':1,
    'max_grad_norm': 1.,
    'tau' : 1e-3,
    'lr': 1e-4,
    'v_optimizer': optim.Adam,
    'lr_v' : 5e-3,
    'loss_func_v': MSELoss,
    'act_func': F.relu,
    'out_act_func': F.tanh,
}
