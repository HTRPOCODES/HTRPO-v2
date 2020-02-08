# These codes are used to preprocess RL envs.
# All borrowed from OppenAI baselines.
import numpy as np
from abc import ABC, abstractmethod
from abc import abstractmethod
from .rms import RunningMeanStd
from collections import OrderedDict
import gym
from gym import spaces
import contextlib
import os
import multiprocessing as mp
import warnings

def space_dim(space):
    if isinstance(space, spaces.Box):
        return space.shape[0] if len(space.shape) == 1 else space.shape
    elif isinstance(space, spaces.MultiBinary):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        return space.nvec.size
    elif isinstance(space, spaces.Dict):
        return {"n_s": space_dim(space.spaces['observation']), "n_g": space_dim(space.spaces['desired_goal'])}
    else:
        raise RuntimeError("Invalid environment observation space. Check and add it in codes manually.")

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape((H, W, h, w, c))
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    return np.stack(obs)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict

def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes

def copy_obs_dict(obs):
    """
    Deep-copy an observation dict.
    """
    return {k: np.copy(v) for k, v in obs.items()}

@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

def obs_to_dict(obs):
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space, max_steps = None, compute_reward = None):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_steps = max_steps
        self.compute_reward = compute_reward

    @abstractmethod
    def reset(self, i = None):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        obs, rews, dones ,infos = self.step_wait()

        for e, info in enumerate(infos):
            if dones[e]:
                if isinstance(obs, dict):
                    infos[e]["new_obs"] = {}
                    temp_obs = self.reset(e)
                    for k in obs.keys():
                        infos[e]["new_obs"][k] = temp_obs[k][e]
                else:
                    infos[e]["new_obs"] = self.reset(e)[e]

        return obs, rews, dones ,infos

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None, max_steps=None):
        self.venv = venv
        VecEnv.__init__(self,
                        num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space,
                        max_steps = max_steps or venv.max_episode_steps,
                        compute_reward=venv.compute_reward if hasattr(venv, "compute_reward") else None)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self, i = None):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space, env.max_episode_steps,
                        env.compute_reward if hasattr(env, "compute_reward") else None)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        if hasattr(env, "spec"):
            self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)

            self._save_obs(e, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self, i = None):
        if i is None:
            for e in range(self.num_envs):
                obs = self.envs[e].reset()
                self._save_obs(e, obs)
        elif isinstance(i, int) and i >= 0 and i < len(self.envs):
            obs = self.envs[i].reset()
            self._save_obs(i, obs)
        else:
            raise RuntimeError("Must indicate a valid index for resetting environment.")
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, act=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):

        VecEnvWrapper.__init__(self, venv)

        self.norm_act = act
        # if action space is not continuous or continuous but unbounded, disable the action normalization.
        if act and (not isinstance(self.venv.action_space, spaces.Box) or
                    (isinstance(self.venv.action_space, spaces.Box)
                     and np.isinf(self.venv.action_space.high - self.venv.action_space.low).sum() > 0)):
            warnings.warn("Only finite continuous action space supports normalization. Action normalization disabled.")
            self.norm_act = False
        self.norm_ob = ob
        self.norm_rw = ret

        self.a_mean = None
        self.a_std = None

        # Different from observation and reward normalization, action normalization is to map actions to [-1,1]
        if isinstance(self.observation_space, spaces.Dict):
            self.ob_rms = {}
            for key in self.observation_space.spaces.keys():
                self.ob_rms[key] = RunningMeanStd(shape=self.observation_space.spaces[key].shape)
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())

        self.clipob = clipob
        self.cliprew = cliprew

        # return
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, a):
        if self.norm_act:
            # clip and normalize actions.
            a = np.clip(self._unnorm_action(a), self.action_space.low, self.action_space.high)
        return VecEnvWrapper.step(self, a)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        # Normalize rewards
        self.ret_rms.update(self.ret)
        if self.norm_rw:
            # r_out = r / sigma_ret (within [-cliprew, cliprew]) WHY????
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        # reset all returns, which are corresponding to envs that are done for the current episode, to 0.
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        # Normalize obs. (Minus mean and divided by sqrt(var).)
        if isinstance(self.observation_space, spaces.Dict):
            for key in obs.keys():
                self.ob_rms[key].update(obs[key])
        else:
            self.ob_rms.update(obs)

        if self.norm_ob:
            if isinstance(self.observation_space, spaces.Dict):
                for key, value in obs.items():
                    obs[key] = np.clip(
                        (obs[key] - self.ob_rms[key].mean) / np.sqrt(self.ob_rms[key].var + self.epsilon), -self.clipob,
                        self.clipob)
            else:
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def _unnorm_action(self, normed_action):
        a_mean = self.a_mean \
            if self.a_mean is not None \
            else 0.5 * (self.action_space.low + self.action_space.high)
        a_std = self.a_std \
            if self.a_std is not None \
            else 0.5 * (self.action_space.high - self.action_space.low)
        return normed_action * a_std + a_mean

    def reset(self, i = None):
        self.ret = np.zeros(self.num_envs)
        if i is not None:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset(i)
        return self._obfilt(obs)

class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
