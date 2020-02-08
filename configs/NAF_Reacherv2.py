from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss

NAFconfig = {
    'memory_size': 100000,
    'batch_size': 4096,
    'reward_decay': 0.95,
    'noise_var' : 1.,
    'noise_min' : 0.,
    'noise_decrease' : 0.0001,
    'optimizer': optim.Adam,
    'loss': MSELoss,
    'action_bounds':1,
    'max_grad_norm': 1,
    'tau' : 0.001,
    'lr':0.001,
}
