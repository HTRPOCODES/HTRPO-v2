import os
import shutil
import uuid
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gym
from gym import spaces

try:
    from atari_py import ALEInterface
except Exception as e:
    print("Could not load ale_python_interface. Pacman's not available.")

"""
This is modified freely from the OpenAI Gym wrapper around atari 
https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
"""


class AtariWrapper():
    """
    ALE wrapper that tries to mimic the options in the DQN paper including the
    preprocessing (except resizing/cropping)
    """
    action_words = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]
    _action_set = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    # Valid actions for ALE.
    # Possible actions are just a list from 0,num_valid_actions
    # We still need to map from the latter to the former when

    possible_actions = list(range(len(_action_set)))

    def __init__(self, rom_path, seed=123,
                 frameskip=4,
                 show_display=False,
                 stack_num_states=4,
                 concatenate_state_every=4):
        """

        Parameters:
            Frameskip should be either a tuple (indicating a random range to
            choose from, with the top value exclude), or an int. It's aka action repeat.

            stack_num_states: Number of dimensions/channels to have.

            concatenate_state_every: After how many frames should one channel be appended to state.
                Number is in terms of absolute frames independent of frameskip
        """

        self.stack_num_states = stack_num_states
        self.concatenate_state_every = concatenate_state_every

        self.game_path = rom_path
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist' % (game, self.game_path))
        self.frameskip = frameskip

        try:
            self.ale = ALEInterface()
        except Exception as e:
            print("ALEInterface could not be loaded. ale_python_interface import failed")
            raise e

        # Set some default options
        self.ale.setInt(b'random_seed', seed)
        self.ale.setBool(b'sound', False)
        self.ale.setBool(b'display_screen', show_display)
        self.ale.setFloat(b'repeat_action_probability', 0.)

        # Load the rom
        self.ale.loadROM(self.game_path)

        (self.screen_width, self.screen_height) = self.ale.getScreenDims()
        self.latest_frame_fifo = deque(maxlen=2)  # Holds the two closest frames to max.
        self.state_fifo = deque(maxlen=stack_num_states)

    def _step(self, a, force_noop=False):
        """Perform one step of the environment.
        Automatically repeats the step self.frameskip number of times

        parameters:
            force_noop: Force it to perform a no-op ignoring the action supplied.
        """
        assert a in self.possible_actions + [0]

        if force_noop:
            action, num_steps = 0, 1
        else:
            action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = np.random.randint(self.frameskip[0], self.frameskip[1])

        reward = 0.0
        for i in range(num_steps):
            reward += self.ale.act(action)
            cur_frame = self.observe_raw(get_rgb=True)
            cur_frame_cropped = self.crop_frame(cur_frame)
            self.latest_frame_fifo.append(cur_frame_cropped)

            if i % self.concatenate_state_every == 0:
                curmax_frame = np.amax(self.latest_frame_fifo, axis=0)
                frame_lumi = self.convert_to_gray(curmax_frame)
                self.state_fifo.append(frame_lumi)

        # Transpose so we get HxWxC instead of CxHxW
        self.current_frame = np.array(np.transpose(self.state_fifo, (1, 2, 0)))
        self.current_frame = cv2.resize(self.current_frame, (84, 84))
        return self.current_frame, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def step(self, *args, **kwargs):
        """Performs one step of the environment
        """
        lives_before = self.ale.lives()
        next_state, reward, done, info = self._step(*args, **kwargs)
        lives_after = self.ale.lives()

        # End the episode when a life is lost
        if lives_before > lives_after:
            done = True

        return next_state, reward, done, info

    def observe_raw(self, get_rgb=False):
        """Observe either RGB or Gray frames.
        Initialzing arrays forces it to not modify stale pointers
        """
        if get_rgb:
            cur_frame_rgb = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            self.ale.getScreenRGB(cur_frame_rgb)
            return cur_frame_rgb
        else:
            cur_frame_gray = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
            self.ale.getScreenGrayscale(cur_frame_gray)
            return cur_frame_gray

    def crop_frame(self, frame):
        """Simply crops a frame. Does nothing by default.
        """
        return frame

    def convert_to_gray(self, img):
        """Get Luminescence channel
        """
        img_f = np.float32(img)
        img_lumi = 0.299 * img_f[:, :, 0] + \
                   0.587 * img_f[:, :, 1] + \
                   0.114 * img_f[:, :, 2]
        return np.uint8(img_lumi)

    def reset(self):
        """Reset the game
        """
        self.ale.reset_game()
        s = self.observe_raw(get_rgb=True)
        s = self.crop_frame(s)

        # Populate missing frames with blank ones.
        for _ in range(self.stack_num_states - 1):
            self.state_fifo.append(np.zeros(shape=(s.shape[0], s.shape[1])))

        self.latest_frame_fifo.append(s)

        # Push the latest frame
        curmax_frame = s
        frame_lumi = self.convert_to_gray(s)
        self.state_fifo.append(frame_lumi)

        self.state = np.transpose(self.state_fifo, (1, 2, 0))
        self.state = cv2.resize(self.state, (84, 84))
        return self.state

    def get_action_meanings(self):
        """Return in text what the actions correspond to.
        """
        return [ACTION_MEANING[i] for i in self._action_set]

    def save_state(self):
        """Saves the current state and returns a identifier to saved state
        """
        return self.ale.cloneSystemState()

    def restore_state(self, ident):
        """Restore game state
        Restores the saved state of the system and perform a no-op
        so a new frame can be generated incase a restore is followed
        by an observe()
        """

        self.ale.restoreSystemState(ident)
        self.step(0, force_noop=True)


class LivePlotter():
    def __init__(self, data, dtype="img", blit=True, save_video_path=None):
        import matplotlib
        plt.switch_backend('TkAgg')
        # import matplotlib
        # matplotlib.use('TkAgg')
        print("Live Plotter initialized using backend", matplotlib.get_backend())

        self.fig = plt.figure(figsize=(5, 7))
        self.ax = self.fig.gca()
        self.blit = blit  # Fast Drawing

        self.dtype = dtype
        self.frame_number = 0

        if self.blit:
            self.fig.canvas.draw()

        if dtype == "img":

            cmap = "gray" if len(data.shape) == 3 else None
            self.h1 = plt.imshow(data, cmap=cmap)

        else:
            self.h1, = plt.plot(data)

        plt.tight_layout()
        if self.blit:
            # fig.canvas.draw()
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.save_video_path = save_video_path

        if save_video_path is not None:
            # Create a folder in temp to dump frames.
            self.frame_dump_folder = "/tmp/" + uuid.uuid4().hex
            print("Creating directory ", self.frame_dump_folder)
            os.mkdir(self.frame_dump_folder)

        self.redraw_figure()

    def update(self, data):
        if self.dtype == "img":
            self.h1.set_data(data)
            self.frame_number += 1

            if self.save_video_path is not None:
                fname = self.frame_dump_folder + "/"
                fname += "frame" + str(self.frame_number) + ".png"
                # print("Saving figure to ",fname)
                plt.savefig(fname)

        else:

            self.h1.set_xdata(list(range(len(data))))
            self.h1.set_ydata(data)

            self.ax.set_xlim(0, len(data))
            maxd = max(data)
            self.ax.set_ylim(0, int(maxd * 1.1))

        self.redraw_figure()

    def redraw_figure(self):
        if self.blit:
            self.fig.canvas.restore_region(self.axbackground)

            # redraw just the points
            self.ax.draw_artist(self.h1)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)
            # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # it is mentionned that blit causes strong memory leakage.
            # however, I did not observe that.

        self.fig.canvas.flush_events()
        plt.show(block=False)


class MsPacman(AtariWrapper):
    """
    Modifications from normal pacman:

    Action Space limited to [up,down,left,right]. No-op used internally but not exposed.
    On reset, we skip 260 frames. The game is essentially frozen and no actions change anything.
    Overlay a 14x19 grid on top and look for pacman
    """
    action_words = ['UP', 'RIGHT', 'LEFT', 'DOWN']
    _action_set = [2, 3, 4, 5]
    possible_actions = list(range(len(_action_set)))

    # Sets game mode to only return (r,c) as observations.
    # This is only to sanity check the network *can* solve it
    only_locations = False

    d_observations = (84, 84, 4) if not only_locations else 2
    d_goals = 2
    n_actions = 4

    # Maintain list of reachable goals
    is_valid_goal = np.array([
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.bool)
    valid_goals = np.transpose(np.nonzero(is_valid_goal))

    def __init__(self, rom_path=b"/data0/svc4/ms_pacman.bin",
                 randomstart=False,
                 max_steps=26,
                 reward='sparse'):

        AtariWrapper.__init__(self, rom_path,
                              frameskip=13,
                              stack_num_states=4,
                              concatenate_state_every=4,
                              )

        self.max_episode_steps = max_steps
        self.reward_type = reward
        self.acc_rew = 0

        self.all_saved_states = self.generate_saved_states(randomstart)

        # The following is needed for live plotting/rendering with the goal location drawn.
        self.live_display = None
        self.loc_pixel_lookup = {}  # Translate goal space to pixel space range
        prev_row = 0
        for row_i, row in enumerate(range(12, 170, 12)):
            prev_col = 0
            for col_i, col in enumerate(
                    [10, 20, 28, 36, 44, 52, 60, 68, 76, 80, 88, 96, 104, 112, 120, 128, 136, 144, -1]):
                self.loc_pixel_lookup[(row_i, col_i)] = (prev_row, row, prev_col, col)
                prev_col = col
            prev_row = row

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(0 * np.ones(self.d_observations), 255 * np.ones(self.d_observations)),
            "desired_goal": spaces.Box(- np.inf * np.ones(self.d_goals), np.inf * np.ones(self.d_goals)),
            "achieved_goal": spaces.Box(- np.inf * np.ones(self.d_goals), np.inf * np.ones(self.d_goals)),
        })

        self.action_space = spaces.Discrete(self.n_actions)

    def generate_saved_states(self, randomstart):
        """
        Play out a list of trajectories so we can collect a bunch of saved states.
        This is needed because frameskipping 300 odd frames every reset
        for MsPacman (which has a period at the beginning where no actions count)
        is a waste of time.

        # trajectories = [[2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 1, 3, 3, 3, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 1, 1, 1], [2, 2, 2, 2, 3, 3, 3, 1, 1, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0], [2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 0, 2, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 1, 1, 1, 3, 1, 1, 3, 3, 3, 0, 1], [2, 3, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 3, 3, 2, 2, 0, 3, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 2, 2, 0, 0], [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 3, 0, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 1, 1, 2, 3, 3, 2, 2, 2, 3, 3]]

        5 trajectories that take pacman to the 4 corners and one vertically up.
        Play out the trajectories and collect the states along them so random restarts can pick one of them.
        """

        if randomstart == True:
            raise NotImplementedError()  # Not implemented for frameskip version.
            # 4 trajectories ignoring the one going up in the docstring
            trajectories = [[2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0],
                            [1, 1, 3, 3, 3, 1, 1, 0, 0, 0, 1, 1, 3, 3, 1, 1, 1, 3, 3, 3, 2],
                            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 3],
                            [2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1]]
        else:
            # Trajectory to bump into a wall before control begins.
            # There will be a random action picked after this so it can't completely memorize.
            trajectories = [[0]]

        all_saved_states = []
        for trajectory in trajectories:
            self.reset(pick_random_state=False)
            for action in trajectory:
                _ = self.step(action, ignore_goal=True)
                all_saved_states += [self.save_state()]
        print("Generated", len(all_saved_states), " start states ")
        return all_saved_states

    def reset(self, pick_random_state=True):

        if pick_random_state:
            _ = AtariWrapper.reset(self)

            random_state_index = np.random.randint(0, len(self.all_saved_states))
            self.restore_state(self.all_saved_states[random_state_index])

            # Then perform a random action
            raction = np.random.randint(0, 4)
            _ = self.step(raction)

        else:
            _ = AtariWrapper.reset(self)
            # Do 250 frame skips because nothing happens at the beginning and actions are pointless.
            # This number is bigger than DQN's 30 no-op_max (30 *(action_repeat: Default 4))
            for i in range(240):
                # Bypass our act and perform it on ALE directly. No Frameskip this way.
                self.ale.act(2)

        self.n_steps = 0
        self.cur_grid_loc = self.find_pacman(self.observe_raw(get_rgb=False))

        goal_index = np.random.randint(0, len(self.valid_goals))
        self.goal = self.valid_goals[goal_index]
        self.acc_rew = 0
        self.achieved_goal = self.cur_grid_loc

        # If it's in debug mode and only training on locations.
        if self.only_locations:
            self.state = self.achieved_goal

        obs = {}
        obs["observation"] = np.array(self.state) / 255
        obs["achieved_goal"] = np.array(self.achieved_goal)
        obs["desired_goal"] = np.array(self.goal)

        return obs

    def crop_frame(self, frame):
        """Crops a frame  This is a divergence from DQN's options.
        """

        return frame[2:170, 2:-2, :]

    def find_pacman(self, frame):
        """Finds the(x,y) location of pacman in the overlaid grid.
        """
        # Overlay a grid and find the count of yellow(bright) pixels present in the region.
        maxpixels, maxpixels_loc = 0, (-1, -1)
        prev_row = 0
        for row_i, row in enumerate(range(12, 170, 12)):
            prev_col = 0
            for col_i, col in enumerate([10, 20, 28, 36, 44, 52, 60,
                                         68, 76, 80, 88, 96, 104, 112, 120,
                                         128, 136, 144, -1]):
                pacman_pixel_count = np.sum(np.logical_and(frame[prev_row:row, prev_col:col] > 160
                                                           , frame[prev_row:row, prev_col:col] < 170))
                if maxpixels < pacman_pixel_count:
                    maxpixels = pacman_pixel_count
                    maxpixels_loc = (row_i, col_i)

                    if (maxpixels / ((row - prev_row) * (col - prev_col))) > 0.4:
                        # A small early exit.
                        # 0.4 seems arbitrary but pacman already
                        # doesn't occupy the whole square and this was
                        # enough empirically.
                        return maxpixels_loc

                prev_col = col
            prev_row = row
        return maxpixels_loc

    def step(self, a, force_noop=False, ignore_goal=False):
        """
        ignore_goal: Flag used when we're traversing trajectories
                    for start states
        """
        lives_before = self.ale.lives()
        if isinstance(a, int):
            self.state, reward, done, info = self._step(a, force_noop=force_noop)
        else:
            self.state, reward, done, info = self._step(a.squeeze(), force_noop=force_noop)
        lives_after = self.ale.lives()

        if force_noop:
            return self.state, None, None

        self.n_steps += 1

        self.new_grid_loc = self.find_pacman(self.observe_raw(get_rgb=False))

        # Check if we've reached goal
        if self.reward_type == 'sparse':
            if np.allclose(self.new_grid_loc, self.goal) and not ignore_goal:
                reward = 0.0
            else:
                reward = -1.0
        else:
            reward = - np.abs(np.array(self.new_grid_loc) - np.array(self.goal)).sum()
        self.acc_rew += reward

        done = (self.max_episode_steps <= self.n_steps) or (reward == 0.0) or (lives_before > lives_after)

        info['is_success'] = (reward == 0.0)
        if done:
            info["episode"]={
                "l" : self.n_steps,
                "r" : self.acc_rew
            }
            self.acc_rew = 0

        self.achieved_goal = self.new_grid_loc

        # If it's in test mode without visual input,
        if self.only_locations:
            self.state = self.achieved_goal

        obs = {}
        obs["observation"] = np.array(self.state) / 255
        obs["achieved_goal"] = np.array(self.achieved_goal)
        obs["desired_goal"] = np.array(self.goal)

        return obs, reward, done, info

    def subgoals(self, episodes, subgoals_per_episode):

        goals = [e.achieved_goals for e in episodes]
        if subgoals_per_episode > 0:
            goals = []
            for e in episodes:
                observations_ep = np.unique(e.achieved_goals, axis=0)

                size = min(subgoals_per_episode, observations_ep.shape[0])
                indices = np.random.choice(observations_ep.shape[0], size, False)
                goals.append(observations_ep[indices])

        uniq_goals = np.unique(np.concatenate(goals, axis=0), axis=0)
        return uniq_goals[uniq_goals[:, 0] != -1, :]

    def compute_reward(self, achieved_goal, desired_goal, info = None):
        assert achieved_goal.shape == desired_goal.shape
        if self.reward_type == "dense":
            return -np.abs((achieved_goal - desired_goal)).sum(axis=-1)
        else:
            # return: dist Ng x T
            dif = np.abs((achieved_goal - desired_goal)).sum(axis=-1)
            return - (dif > 0).astype(np.float32)

    def render(self, **args):
        return
        if "vizualise" in args and args["vizualise"]:
            self.env.render_mpl(*args)
        else:
            print(self.__repr__())

    def seed(self, seed):
        pass

    def render_mpl(self):
        current_frame_col = self.observe_raw(get_rgb=True)
        if self.live_display is None:
            # plt.switch_backend('MacOSX')
            self.live_display = LivePlotter(current_frame_col)

        raw_frame = np.array(self.observe_raw(get_rgb=True))
        goal_pixels = self.loc_pixel_lookup[tuple(self.goal)]

        raw_frame[(goal_pixels[0] + goal_pixels[1]) / 2, goal_pixels[2]:goal_pixels[3], :] = 255
        self.live_display.update(raw_frame)

print("Successful!!!!")
