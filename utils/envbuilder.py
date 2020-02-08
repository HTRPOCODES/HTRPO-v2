
from .vecenv import DummyVecEnv, VecNormalize, VecFrameStack
from collections import defaultdict
import gym
from gym.wrappers import FlattenDictWrapper
try:
    import robosuite as robst
except Exception as e:
    "Could not find package robosuite. All relevant environments cannot be used."
import multiprocessing
import sys
import re
from .atariwrapper import make_atari, wrap_deepmind
import numpy as np
import random
import envs
from gym import spaces
from .monitor import Monitor

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

HER_ENV_LIST = {"FlipBit8", "FlipBit16", "FlipBit32", "FlipBit48", "EmptyMaze", "FourRoomMaze", "FetchPushDiscrete",
                "FetchReachDiscrete", "FetchSlideDiscrete", "MsPacman"}
my_own_envs = {"BaxterReacher-v0", "FlipBit8", "FlipBit16", "FlipBit32", "FlipBit48", "EmptyMaze", "FourRoomMaze",
               "FetchPushDiscrete", "FetchReachDiscrete", "FetchSlideDiscrete", "AntReacher", "Reacher",
               "MsPacman"}
robotsuite_envs = {"BaxterLift", "BaxterPegInHole", "SawyerLift", "SawyerNutAssembly", "SawyerNutAssemblyRound",
                   "SawyerNutAssemblySingle", "SawyerNutAssemblySquare", "SawyerPickPlace", "SawyerPickPlaceBread",
                   "SawyerPickPlaceCan", "SawyerPickPlaceCereal", "SawyerPickPlaceMilk", "SawyerPickPlaceSingle",
                   "SawyerStack"}

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def get_env_type(args):
    env_id = args.env

    if env_id in my_own_envs:
        return "myown", env_id
    elif env_id in robotsuite_envs:
        return "robotsuite", env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    nenv = args.num_envs or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_id == "BaxterReacher-v0":
        return eval("envs." + "".join(env_id.split("-")) + "()"), env_type, env_id

    if env_type in {'atari'}:
        if alg == 'DQN' or alg == 'DDQN' or alg == 'DuelingDQN' :
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'TRPO':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed)
            env = VecFrameStack(env, frame_stack_size)
    else:
        flatten_dict_observations = alg not in {'HTRPO'}
        env = make_vec_env(env_id, env_type, args.num_envs or 1, seed,
                           flatten_dict_observations=flatten_dict_observations, render = args.render)

        if env_type in {'mujoco', 'robotics', 'robotsuite'} :
           env = VecNormalize(env, ob=not args.unnormobs, ret=not args.unnormret, act=not args.unnormact)

    return env, env_type, env_id

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 flatten_dict_observations=True,
                 render = False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            subrank=rank,
            seed=seed,
            wrapper_kwargs=wrapper_kwargs,
            flatten_dict_observations=flatten_dict_observations,
            render = render
        )

    set_global_seeds(seed)

    if num_env > 1:
        return DummyVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, env_type, subrank=0, seed=None, wrapper_kwargs=None,
             flatten_dict_observations = True, render = False):
    wrapper_kwargs = wrapper_kwargs or {}
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'myown':
        env = eval("envs." + "".join(env_id.split("-")) + "()")
    elif env_type == 'robotsuite':
        env = eval("envs." + "".join(env_id.split("-")) + "(render = render)")
    else:
        env = gym.make(env_id)
        env.max_episode_steps = env.spec.max_episode_steps

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    if env_id not in my_own_envs and env_id not in robotsuite_envs:
        env = Monitor(env, allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)

    return env

def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    env.seed(seed)

    return env

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import torch
        torch.manual_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)

def init_her_env(env):
    pass