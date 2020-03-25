from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F

AGENT_CONFIG = {
    # 'lr' indicates the learning rate of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'lr':0.01,
    'mom':None,
    'reward_decay':0.9,
    'memory_size': 1e6,
    # 'hidden_layers' defines the layers of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'hidden_layers':[64, 64],
    'act_func': F.tanh,
    'out_act_func': None,
    'using_bn': False,
}

DQN_CONFIG = {
    'replace_target_iter':600,
    'e_greedy':0.9,
    'e_greedy_increment':None,
    'optimizer': optim.RMSprop,
    'loss' : MSELoss,
    'batch_size': 32,
}

DDPG_CONFIG = {
    'steps_per_iter': 50,
    'learn_start_step': 10000,
    'batch_size': 128,
    'reward_decay': 0.99,
    'tau' : 0.005,
    'noise_var' : 0.3,
    'noise_min' : 0.01,
    'noise_decrease' : 2e-5,
    'optimizer': optim.Adam,
    'v_optimizer': optim.Adam,
    'lr': 1e-4,
    'lr_v' : 1e-3,
    'hidden_layers': [400, 300],
    'hidden_layers_v' : [400, 300],
    'loss_func_v': MSELoss,
    'act_func': F.relu,
    'out_act_func': F.tanh,
    'action_bounds':1,
    'max_grad_norm': None,
}

TD3_CONFIG = {
    'actor_delayed_steps': 2,
    'smooth_epsilon': 0.5,
    'smooth_noise': 0.2,
}

NAF_CONFIG = {
    'steps_per_iter': 50,
    'learn_start_step': 10000,
    'tau' : 0.005,
    'lr' : 1e-3,
    'noise_var' : 0.3,
    'noise_min' : 0.01,
    'noise_decrease' : 2e-5,
    'optimizer': optim.Adam,
    'loss': MSELoss,
    'batch_size': 128,
    'hidden_layers': [400, 300],
    'action_bounds':1,
    'max_grad_norm': 1.,
    'act_func': F.tanh,
    'using_bn': True,
}

PG_CONFIG = {
    'max_grad_norm': 2,
    'steps_per_iter': 2048,
    'action_bounds': 1,
    'optimizer':optim.Adam,
    'GAE_lambda' : 0.95,            # HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. 2016 ICLR
    'entropy_weight':0.0,
    'init_noise': 1.,
    'value_type': 'FC',
    'hidden_layers_v' : [64,64],
    'loss_func_v':MSELoss,
    'v_optimizer': optim.Adam,
    'mom_v' : None,
    'lr_v' : 0.01,
    'iters_v': 3,
    'using_KL_estimation' : False,
    'policy_type': 'FC',
}
PG_CONFIG['memory_size'] = PG_CONFIG['steps_per_iter']

NPG_CONFIG = {
    'cg_iters': 10,
    'cg_residual_tol' : 1e-10,
    'cg_damping': 1e-3,
    'max_kl_divergence':0.01,
}

PPO_CONFIG = {
    'nbatch_per_iter': 32,
    'updates_per_iter': 10,
    'clip_epsilon': 0.2,
    'lr': 3e-4,
    'v_coef': 0.5,
}
PPO_CONFIG['lr_v'] = PPO_CONFIG['lr']

AdaptiveKLPPO_CONFIG = {
    'init_beta':3.,
    'nbatch_per_iter': 32,
    'updates_per_iter': 10,
    'lr': 3e-4,
    'v_coef': 0.5,
}
AdaptiveKLPPO_CONFIG['lr_v'] = AdaptiveKLPPO_CONFIG['lr']

TRPO_CONFIG = {
    'max_search_num' : 10,
    'accept_ratio' : .1,
    'step_frac': .5
}

HTRPO_CONFIG = {
    'hindsight_steps': 10,
    'sampled_goal_num': 10,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
    'using_hgf_goals' : True,
    'using_KL_estimation' : True,
    'using_hpg': False,
    'using_original_data': False,
    'KL_esti_method_for_TRPO' : 'kl2',
}