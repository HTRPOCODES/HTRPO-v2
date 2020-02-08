import re
import numpy as np
from scipy.sparse.csgraph import shortest_path
from gym import spaces

class Maze(object):
    def __init__(self, layout, max_steps, entries, exits=None, epsilon=0.0, reward = 'sparse'):
        self.layout = np.array(layout, dtype=np.int)
        validr, validc = np.nonzero(self.layout)
        self.valid_positions = set(zip(validr, validc))

        self.entries = set(entries)

        self.exits = self.valid_positions - self.entries
        if exits is not None:
            self.exits = set(exits)

        self.epsilon = epsilon

        self.check_consistency()
        self.compute_distance_matrix()

        self.n_actions = 4
        self.d_observations = 2
        self.d_goals = 2

        self.reward_type = reward

        self.acc_rew = 0

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "observation": spaces.MultiDiscrete([self.layout.shape[0], self.layout.shape[1]]),
            "desired_goal": spaces.MultiDiscrete([self.layout.shape[0], self.layout.shape[1]]),
            "achieved_goal": spaces.MultiDiscrete([self.layout.shape[0], self.layout.shape[1]]),
        })

        self.max_episode_steps = max_steps

    def check_consistency(self):
        given = self.entries.union(self.exits)

        if not given.issubset(self.valid_positions):
            raise Exception('Invalid entry or exit.')

        if len(self.entries.intersection(self.exits)) > 0:
            raise Exception('Entries and exits must be disjoint.')

    def compute_distance_matrix(self):
        shape = self.layout.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        adj_matrix = np.zeros((self.layout.size, self.layout.size))

        for (r, c) in self.valid_positions:
            index = np.ravel_multi_index((r, c), shape)

            for move in moves:
                nr, nc = r + move[0], c + move[1]

                if (nr, nc) in self.valid_positions:
                    nindex = np.ravel_multi_index((nr, nc), shape)
                    adj_matrix[index, nindex] = 1

        self.dist_matrix = shortest_path(adj_matrix)

    def distance(self, orig, dest):
        shape = self.layout.shape

        o_index = np.ravel_multi_index((int(orig[0]), int(orig[1])), shape)
        d_index = np.ravel_multi_index((int(dest[0]), int(dest[1])), shape)

        distance = self.dist_matrix[o_index, d_index]
        if not np.isfinite(distance):
            raise Exception('There is no path between origin and destination.')

        return distance

    def reset(self):
        self.acc_rew = 0
        self.n_steps = 0

        i = np.random.choice(len(self.entries))
        self.position = sorted(self.entries)[i]

        i = np.random.choice(len(self.exits))
        self.goal = sorted(self.exits)[i]

        obs = {
            "observation": np.array(self.position),
            "desired_goal": np.array(self.goal),
            "achieved_goal": np.array(self.position).copy(),
        }

        return obs

    def step(self, a):
        """a: up, down, left, right"""
        if a >= self.n_actions:
            raise Exception('Invalid action')

        if np.random.random() < self.epsilon:
            a = [np.random.choice(self.n_actions)]

        self.n_steps += 1

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        newr = self.position[0] + moves[a[0]][0]
        newc = self.position[1] + moves[a[0]][1]

        if (newr, newc) in self.valid_positions:
            self.position = (newr, newc)

        if self.reward_type == "dense":
            reward = -np.abs((np.array(self.position) - np.array(self.goal))).sum() / 20
            # when reaching the goal, an extra reward is added
            if self.position == self.goal:
                reward += 5
        else:
            if self.position == self.goal:
                reward = 0.0
            else:
                reward = -1.0
        self.acc_rew += reward

        done = (self.max_episode_steps <= self.n_steps) or (reward >= 0.0)

        obs = {
            "observation": np.array(self.position),
            "desired_goal": np.array(self.goal),
            "achieved_goal": np.array(self.position).copy(),
        }

        info = {'is_success': reward >= 0.0}
        if done:
            info['episode'] = {
                'l': self.n_steps,
                'r': self.acc_rew,
            }

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        # return: dist Ng x T
        dif = np.abs((achieved_goal - desired_goal)).sum(axis=-1)
        if self.reward_type == "dense":
            return -np.abs((achieved_goal - desired_goal)).sum(axis=-1) + 5 * (dif == 0).astype(np.float32)
        else:
            return - (dif > 0).astype(np.float32)

    def seed(self, seed):
        np.random.seed(seed)

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        s = []

        for i in range(len(self.layout)):
            for j in range(len(self.layout[0])):
                if (i, j) == self.position:
                    s.append('@')
                elif (i, j) == self.goal:
                    s.append('$')
                else:
                    s.append('.' if self.layout[i, j] else '#')
            s.append('\n')

        return ''.join(s)

class EmptyMaze(Maze):
    def __init__(self, reward = 'sparse'):
        super(EmptyMaze, self).__init__(layout=np.ones((11, 11), dtype=np.int), max_steps = 32, entries=[(0, 0)],
                                        reward = reward)

class FourRoomMaze(Maze):
    def __init__(self, reward = 'sparse'):
        layout = np.ones(shape=(11, 11), dtype=np.int)

        # Walls
        layout[:, 5] = 0
        layout[5, :5] = 0
        layout[6, 6:] = 0

        # Doors
        layout[5, 1] = 1
        layout[2, 5] = 1
        layout[6, 8] = 1
        layout[9, 5] = 1
        super(FourRoomMaze, self).__init__(layout = layout, max_steps=32,
                                           entries=[(0, 0), (0, 10), (10, 0), (10, 10)],
                                           epsilon=0.2, reward = reward)