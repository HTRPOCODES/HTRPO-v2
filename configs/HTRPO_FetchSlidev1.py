from torch.nn import functional as F

HTRPOconfig = {
    'cg_damping': 1e-3,
    'reward_decay': 0.98,
    'GAE_lambda': 0.,
    'max_kl_divergence': 2e-5,
    'entropy_weight': 1e-4,
    'per_decision': True,
    'weighted_is': True,
    'using_active_goals' : True,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'lr_v': 5e-4,
    'iters_v':20,
    # for comparison with HPG
    'lr': 5e-4,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': True,
    'steps_per_iter': 3200,
    'sampled_goal_num': 100,
    'value_type': 'FC',
    'using_original_data': False,
    'out_act_func': F.tanh,
    'using_kl2': True
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']

