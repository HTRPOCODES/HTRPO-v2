import re
import numpy as np
from scipy.sparse.csgraph import shortest_path
from gym import spaces
class FlipBit(object):
    def __init__(self, n_bits = 8, reward = 'sparse'):
        self.n_actions = n_bits
        self.action_space = spaces.Discrete(n_bits)
        self.d_observations = n_bits
        self.d_goals = n_bits
        self.observation_space = spaces.Dict({
            "observation": spaces.MultiBinary(n_bits),
            "desired_goal": spaces.MultiBinary(n_bits),
            "achieved_goal": spaces.MultiBinary(n_bits),
        })
        self.max_episode_steps = n_bits

        self.reward_type = reward

        self.acc_rew = 0

    def reset(self):
        self.acc_rew = 0
        self.n_steps = 0
        self.state = np.zeros(self.d_observations, dtype=np.int32)
        # self.state = np.random.randint(0, 2, size=self.d_observations)
        self.goal = np.random.randint(0, 2, size=self.d_goals, dtype=np.int32)
        state, goal = np.array(self.state), np.array(self.goal)
        obs = {
            "observation": state,
            "desired_goal": goal,
            "achieved_goal": state.copy(),
        }
        return obs

    def step(self, a):
        if a[0] >= self.n_actions:
            raise Exception('Invalid action')
        self.n_steps += 1
        self.state[a[0]] = 1 - self.state[a[0]]
        if self.reward_type == "dense":
            reward = -np.abs((self.state - self.goal)).sum() / self.n_actions
            # when reaching the goal, an extra reward is added
            if np.allclose(self.state, self.goal):
                reward += 5
        else:
            if np.allclose(self.state, self.goal):
                reward = 0.
            else:
                reward = -1.
        self.acc_rew += reward

        done = (self.max_episode_steps <= self.n_steps) or (reward >= 0.)

        obs = {
            "observation": self.state,
            "desired_goal": self.goal,
            "achieved_goal": self.state.copy(),
        }

        info = {'is_success': reward >= 0}
        if done:
            info['episode'] = {
                'l' : self.n_steps,
                'r' : self.acc_rew,
            }

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        dif = np.abs((achieved_goal - desired_goal)).sum(axis=-1)
        if self.reward_type == "dense":
            return -np.abs((achieved_goal - desired_goal)).sum(axis=-1) + 5 * (dif == 0).astype(np.float32)
        else:
            return - (dif > 0).astype(np.float32)

    def render(self):
        print(self.__repr__())

    def seed(self, seed):
        np.random.seed(seed)

    def __repr__(self):
        return 'State: {0}. Goal: {1}.'.format(self.state, self.goal)

class FlipBit8(FlipBit):
    def __init__(self, reward = 'sparse'):
        super(FlipBit8, self).__init__(n_bits = 8, reward = reward)

class FlipBit16(FlipBit):
    def __init__(self, reward = 'sparse'):
        super(FlipBit16, self).__init__(n_bits = 16, reward = reward)

class FlipBit32(FlipBit):
    def __init__(self, reward = 'sparse'):
        super(FlipBit32, self).__init__(n_bits = 32, reward = reward)

class FlipBit48(FlipBit):
    def __init__(self, reward = 'sparse'):
        super(FlipBit48, self).__init__(n_bits = 48, reward = reward)
