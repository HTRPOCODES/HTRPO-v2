import torch.nn.functional as F

HTRPOconfig = {
    'reward_decay': 0.98,
    'cg_damping': 1e-3,
    'GAE_lambda': 0.,
    'max_kl_divergence': 2e-5,
    'entropy_weight': 0,
    'per_decision': True,
    'weighted_is': True,
    'using_active_goals' : True,
    'hidden_layers': [32, 32, 32],
    'hidden_layers_v': [32, 32, 32],
    'max_grad_norm': None,
    'lr_v': 5e-5,
    'iters_v':10,
    # for comparison with HPG
    'lr': 3e-5,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': False,
    'steps_per_iter': 300,
    'sampled_goal_num': 16,
    'value_type': None,
    'using_original_data': False,
    'using_kl2':True,
    'policy_type':'Conv',
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
