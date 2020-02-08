from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss

PGconfig = {
    'max_grad_norm': 2,
    'steps_per_iter': 2048,
    'action_bounds': 1,
    'optimizer':optim.Adam,
    'value_type' : 'FC',
    'hidden_layers_v': [64, 64],
    'GAE_lambda' : 0.95,            # HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. 2016 ICLR
    'loss_func_v':MSELoss,
    'v_optimizer':optim.Adam,
    'lr': 0.001,
    'lr_v' : 0.001,
    'entropy_weight':0.0,
    'mom_v' : None,
    'init_noise': 1.,
}
PGconfig['memory_size'] = PGconfig['steps_per_iter']