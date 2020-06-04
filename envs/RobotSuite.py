import abc

try:
    import robosuite as suite
    from robosuite.wrappers import *
except Exception as e:
    "Could not find package robosuite. All relevant environments cannot be used."

import os

from utils.vecenv import DummyVecEnv, VecNormalize
import gym
from gym.wrappers import FlattenDictWrapper

from utils.atariwrapper import make_atari, wrap_deepmind
import numpy as np
import random
from gym import spaces
from utils.monitor import Monitor

from collections import OrderedDict

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import time

# MILK_STATE_LIST = list(range(1,18)) + list(range(39,49))
# BREAD_STATE_LIST = list(range(1,11)) + list(range(18,25)) + list(range(39,49))
# CEREAL_STATE_LIST = list(range(1,11)) + list(range(25,32)) + list(range(39,49))
# CAN_STATE_LIST = list(range(1,11)) + list(range(32,49))

class RobotSuiteWrapper(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, env_id,
                 using_gym_wrapper = True,
                 state_list=None,
                 max_steps=50,
                 reward="sparse",
                 using_demo_init=False,
                 demo_path = "demos",
                 render = False):
        """
        :param env_id: env name
        :param using_gym_wrapper: whether using gym wrapper provided by SURREAL
        :param using_demo_init: whether apply initialization of demonstration states
        :param state_list: select some dimension of states for the observation (None means all)
                NOTE: this option is only used when not using gym wrapper (using_gym_wrapper = False)
        """
        self.reward_type = reward

        self.env = suite.make(env_id,
                    has_renderer=render,
                    has_offscreen_renderer=False,
                    use_object_obs=True,
                    use_camera_obs=False,
                    reward_shaping=False if reward == "sparse" else True,
                    control_freq=10,
                    ignore_done=True,
                    )

        if using_demo_init:
            self.env = DemoSamplerWrapper(self.env,
                                          demo_path = os.path.join(demo_path, env_id),
                                          scheme_ratios = [0.5, 0.5])

        self.using_gym_wrapper = using_gym_wrapper
        if using_gym_wrapper:
            self.env = GymWrapper(self.env, keys = state_list
                                                    if state_list is not None else ["robot-state", "object-state"])

        ob, rew, done, _ = self.env.step(np.random.rand(self.env.dof))

        if not using_gym_wrapper:
            # ob = self.env.sim.get_state().flatten()
            self.state_list = state_list
            if self.state_list:
                ob = np.concatenate([ob[i] for i in self.state_list])

        self.max_episode_steps = max_steps
        self.n_steps = 0

        self.n_actions = self.env.dof
        self.d_observations = ob.size
        self.d_goals = self.get_d_goals()

        self.observation_space = spaces.Dict({
            # "observation": self.env.observation_space,
            "observation": spaces.Box(- np.inf * np.ones(self.d_observations), np.inf * np.ones(self.d_observations)),
            "desired_goal": spaces.Box(- np.inf * np.ones(self.d_goals), np.inf * np.ones(self.d_goals)),
            "achieved_goal": spaces.Box(- np.inf * np.ones(self.d_goals), np.inf * np.ones(self.d_goals)),
        })
        self.action_space = self.env.action_space

        self.acc_rew = 0

    def step(self, action):
        self.n_steps += 1
        ob, rew, done, info = self.env.step(action)
        if not self.using_gym_wrapper:
            if self.state_list:
                ob = np.concatenate([ob[i] for i in self.state_list])
            else:
                raise RuntimeError("State list must be specified for training when you are not using gym wrapper.")

        ag = self.read_achieved_goal()
        dg = self.read_desired_goal()
        reached_goal = self.reached_goal(ag, dg)
        done = (self.n_steps >= self.max_episode_steps) or reached_goal or done

        # reward of robosuite is 0 and 1, we should modify it to -1 and 0
        if self.reward_type == "sparse":
            if not reached_goal:
                rew = -1.
            else:
                rew = 0.
        else:
            rew = -np.linalg.norm(ag - dg)
        self.acc_rew += rew

        ob_dict ={
            "observation": ob,
            "achieved_goal": ag,
            "desired_goal":dg
        }

        info = {'is_success': reached_goal}
        if done:
            info['episode'] = {
                'l': self.n_steps,
                'r': self.acc_rew,
            }

        return ob_dict, rew, done, info

    def render(self, mode=None):
        self.env.render()
        # time.sleep(1. / self.env.control_freq)

    def reset(self):
        self.acc_rew = 0

        self.n_steps = 0

        ob = self.env.reset()

        if not self.using_gym_wrapper:
            ob = self.env.sim.get_state().flatten()
            if self.state_list:
                ob = np.array([ob[i] for i in self.state_list])

        ag = self.read_achieved_goal()
        dg = self.read_desired_goal()

        ob_dict ={
            "observation": ob.copy(),
            "achieved_goal": ag.copy(),
            "desired_goal":dg.copy()
        }

        return ob_dict

    def seed(self, seed):
        # TODO: implement random seed initialization for robosuite envs.
        pass

    # for different environments, the compute_reward function should be different
    @abc.abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError("Must be implemented in the subclass.")

    @abc.abstractmethod
    def reached_goal(self, ag, dg):
        raise NotImplementedError("Must be implemented in the subclass.")

    @abc.abstractmethod
    def get_d_goals(self):
        raise NotImplementedError("Must be implemented in the subclass.")

    @abc.abstractmethod
    def read_achieved_goal(self):
        raise NotImplementedError("Must be implemented in the subclass.")

    @abc.abstractmethod
    def read_desired_goal(self):
        raise NotImplementedError("Must be implemented in the subclass.")

class SawyerLift(RobotSuiteWrapper):
    def __init__(self,
                 env_id="SawyerLift",
                 max_steps=50,
                 reward="sparse",
                 render = False):
        super(SawyerLift, self).__init__(env_id, max_steps=max_steps, reward=reward, render = render,
                                         state_list = ["robot-state"])


    # TODO: implement the official reward shaping here
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == "sparse":
            return -(np.linalg.norm(achieved_goal - desired_goal, axis = -1) > 0.05).astype(np.float32)
        else:
            return - np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def reached_goal(self, ag, dg):
        return np.linalg.norm(ag - dg) < 0.05

    def get_d_goals(self):
        return 3

    def read_achieved_goal(self):
        return self.env.sim.data.body_xpos[23]

    def read_desired_goal(self):
        return self.env.sim.data.body_xpos[self.env.cube_body_id]

# class SawyerLift(RobotSuiteWrapper):
#     def __init__(self,
#                  env_id="SawyerLift",
#                  max_steps=50,
#                  reward="sparse",
#                  render = False):
#         super(SawyerLift, self).__init__(env_id, max_steps=max_steps, reward=reward, render = render)
#
#     # TODO: implement the official reward shaping here
#     def compute_reward(self, achieved_goal, desired_goal, info):
#         return -(achieved_goal < desired_goal).squeeze(-1).astype(np.float32)
#
#     def reached_goal(self, ag, dg):
#         return ag[0] >= dg[0]
#
#     def get_d_goals(self):
#         return 1
#
#     def read_achieved_goal(self):
#         return np.array([self.env.sim.data.body_xpos[self.env.cube_body_id][2]])
#
#     def read_desired_goal(self):
#         return np.array([self.env.table_full_size[2] + 0.04])

class SawyerPickPlace(RobotSuiteWrapper):
    def __init__(self,
                 env_id="SawyerPickPlace",
                 state_list = None,
                 max_steps=50,
                 reward="sparse",
                 object_name=None, # "milk"  "bread" "cereal" "can" or None
                 render = False):

        # if all objects are included
        self.all_obj = True if object_name is None else False
        super(SawyerPickPlace, self).__init__(env_id, state_list = state_list,
                                              max_steps=max_steps, reward=reward, render = render)

        # threshold for generating fake data
        self.thresh = 0.05

        # goal settings
        bin_pos = self.env.bin_pos
        bin_size = self.env.bin_size
        # low bounds for 4 bins
        lows = (
            np.array((bin_pos[0] - bin_size[0] / 2, bin_pos[1] - bin_size[1] / 2, bin_pos[2])),
            np.array((bin_pos[0], bin_pos[1] - bin_size[1] / 2, bin_pos[2])),
            np.array((bin_pos[0] - bin_size[0] / 2, bin_pos[1], bin_pos[2])),
            np.array((bin_pos[0], bin_pos[1], bin_pos[2])),
        )
        # high bounds for 4 bins
        highs = (
            np.array((bin_pos[0], bin_pos[1], bin_pos[2] + 0.1)),
            np.array((bin_pos[0] + bin_size[0] / 2, bin_pos[1], bin_pos[2] + 0.1)),
            np.array((bin_pos[0], bin_pos[1] + bin_size[1] / 2, bin_pos[2] + 0.1)),
            np.array((bin_pos[0] + bin_size[0] / 2, bin_pos[1] + bin_size[1] / 2, bin_pos[2] + 0.1)),
        )

        # there is only one object
        if not self.all_obj:
            self.object_id = self.env.object_to_id[object_name]
            # goal space
            self.goal_space = spaces.Box(low=lows[self.object_id], high=highs[self.object_id])
        else:
            self.object_id = self.env.object_to_id.values()
            # goal space
            self.goal_space = spaces.Box(low = np.concatenate(lows), high = np.concatenate(highs[i]))

    # TODO: implement the official reward shaping here
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > self.thresh).astype(np.float32)

    def reached_goal(self, ag, dg):
        if not self.all_obj:
            ret = not self.env.not_in_bin(obj_pos = ag, bin_id = self.object_id)
        else:
            ret = True
            for i in self.object_id:
                ret = ret and self.env.not_in_bin(obj_pos = ag[3*i : 3*i + 3], bin_id = i)
                if not ret:
                    break
        return ret

    def get_d_goals(self):
        if not self.all_obj:
            return 3
        else:
            return 12

    def read_achieved_goal(self):
        if not self.all_obj:
            obj_str = str(self.env.item_names[self.object_id]) + "0"
            obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj_str]]
            return obj_pos
        else:
            obj_poses = np.zeros((0,))
            for obj_id in self.object_id:
                obj_str = str(self.env.item_names[obj_id]) + "0"
                obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj_str]]
                obj_poses = np.concatenate(obj_poses, obj_pos)
            return obj_poses

    def read_desired_goal(self):
        if self.n_steps == 0:
            self.tgt = self.goal_space.sample()
        return self.tgt

class SawyerPickPlaceCereal(SawyerPickPlace):
    def __init__(self,
                 env_id="SawyerPickPlaceCereal",
                 max_steps=50,
                 reward="sparse",
                 object_name="cereal",
                 render = False):
        super(SawyerPickPlaceCereal, self).__init__(env_id=env_id,
                                                    state_list=None,
                                                    max_steps=max_steps,
                                                    reward=reward,
                                                    object_name=object_name,
                                                    render = render)

class SawyerPickPlaceCan(SawyerPickPlace):
    def __init__(self,
                 env_id="SawyerPickPlaceCan",
                 max_steps=50,
                 reward="sparse",
                 object_name="can",
                 render = False):
        super(SawyerPickPlaceCan, self).__init__(env_id=env_id,
                                                state_list=None,
                                                max_steps=max_steps,
                                                reward=reward,
                                                object_name=object_name,
                                                render=render)

class SawyerPickPlaceMilk(SawyerPickPlace):
    def __init__(self,
                 env_id="SawyerPickPlaceMilk",
                 max_steps=50,
                 reward="sparse",
                 object_name="milk",
                 render = False):
        super(SawyerPickPlaceMilk, self).__init__(env_id=env_id,
                                                 state_list=None,
                                                 max_steps=max_steps,
                                                 reward=reward,
                                                 object_name=object_name,
                                                 render=render)

class SawyerPickPlaceBread(SawyerPickPlace):
    def __init__(self,
                 env_id="SawyerPickPlaceBread",
                 max_steps=50,
                 reward="sparse",
                 object_name="bread",
                 render = False):
        super(SawyerPickPlaceBread, self).__init__(env_id=env_id,
                                                  state_list=None,
                                                  max_steps=max_steps,
                                                  reward=reward,
                                                  object_name=object_name,
                                                  render=render)

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 flatten_dict_observations=True):
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
        )

    set_global_seeds(seed)

    if num_env > 1:
        return DummyVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])

def make_env(env_id, env_type, subrank=0, seed=None, wrapper_kwargs=None, flatten_dict_observations = True):
    env = eval("".join(env_id.split("-")) + "()")
    env.seed(seed + subrank if seed is not None else None)
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

if __name__ == "__main__":
    nenv = 1
    seed = 1
    env = make_vec_env("SawyerPickPlaceCan", "robosuite", 1, seed, False)

    for j in range(10):
        obs = env.reset()
        for i in range(500):
            action = np.random.rand(1, 8)
            obs, ret, done, _ = env.step(action)
            # for i,d in enumerate(done):
            #     print(d)
            #     if d:
            #         env.envs[i].reset()
