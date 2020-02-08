HTRPOconfig = {
    'reward_decay': 0.95,
    'max_kl_divergence': 0.001,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
    'GAE_lambda': 0.,
    'using_active_goals' : True,
    'hidden_layers': [64,64],
    'hidden_layers_v': [64,64],
    'max_grad_norm': None,
    'lr_v': 5e-4,
    'iters_v':10,
    # for comparison with HPG
    'lr': 0.001,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': False,
    'steps_per_iter': 256,
    'sampled_goal_num': None,
    'value_type': 'FC',
    'using_original_data': False,
    'using_kl2':True,
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
