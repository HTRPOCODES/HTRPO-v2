from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss

NPGconfig = {
    'cg_iters': 10,
    'cg_residual_tol' : 1e-10,
    'cg_damping': 1e-3,
    'max_kl_divergence':0.01,
    'GAE_lambda' : 0.98,
    'value_type' : 'FC',
    'hidden_layers_v': [64, 64],
    'lr_v': 0.001,
    'iters_v':3,
    'v_optimizer':optim.Adam,
    'reward_decay': 0.99,
    'steps_per_iter': 1024,
}
NPGconfig['memory_size'] = NPGconfig['steps_per_iter']