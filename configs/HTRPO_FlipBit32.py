HTRPOconfig = {
    'cg_damping': 1e-2,
    'GAE_lambda':0.,
    'reward_decay': 0.97,
    'max_kl_divergence': 0.001,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
    'using_active_goals' : True,
    'hidden_layers': [256],
    'hidden_layers_v': [256],
    'max_grad_norm': None,
    'lr_v': 5e-4,
    'iters_v':10,
    # for comparison with HPG
    'lr': 1e-3,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': False,
    'steps_per_iter': 512,
    'sampled_goal_num': None,
    'value_type': 'FC',
    'using_original_data': False,
    'using_kl2':True
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
