# parant class for all agent
import torch
import abc
import copy
from .config import AGENT_CONFIG
from collections import deque

class Agent:
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(AGENT_CONFIG)
        config.update(hyperparams)
        self.n_states = config['n_states']

        # self.n_actions = 0
        if 'n_actions' in config.keys():
            self.n_actions = config['n_actions']
        self.n_action_dims = config['n_action_dims']
        self.lr = config['lr']
        self.mom = config['mom']
        self.gamma = config['reward_decay']
        self.memory_size = config['memory_size']
        self.hidden_layers = config['hidden_layers']
        self.act_func = config['act_func']
        self.out_act_func = config['out_act_func']
        self.dicrete_action = config['dicrete_action']
        self.using_bn = config['using_bn']

        self.norm_ob = None

        self.learn_step_counter = 0
        self.episode_counter = 0

        # used in HTRPO. In other algorithms, it will be set to 0.
        self.max_steps = 0

        self.cost_his = []

        self.use_cuda = False
        self.r = torch.Tensor(1)
        self.done = torch.Tensor(1)
        self.s_ = torch.Tensor(1)
        self.s = torch.Tensor(1)
        self.a = torch.Tensor(1)
        self.logpac_old = torch.Tensor(1)
        self.other_data = None

    @abc.abstractmethod
    def choose_action(self, s, other_data = None, greedy = False):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def store_transition(self, transition):
        self.memory.store_transition(transition)

    def sample_batch(self, batch_size = None):
        return self.memory.sample_batch(batch_size)

    def soft_update(self, target, eval, tau):
        for target_param, param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) +
                                    param.data * tau)

    def hard_update(self, target, eval):
        target.load_state_dict(eval.state_dict())
        # print('\ntarget_params_replaced\n')

    def cuda(self):
        self.use_cuda = True
        self.r = self.r.cuda()
        self.a = self.a.cuda()
        self.s = self.s.cuda()
        self.s_ = self.s_.cuda()
        self.done = self.done.cuda()
        self.logpac_old = self.logpac_old.cuda()

    @abc.abstractmethod
    def save_model(self, save_path):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def load_model(self, load_path, load_point):
        raise NotImplementedError("Must be implemented in subclass.")

    def eval_brain(self, env, render=True, eval_num=None):
        eprew_list = deque(maxlen=eval_num)
        success_history = deque(maxlen=eval_num)
        self.policy = self.policy.eval()
        if self.value is not None:
            self.value = self.value.eval()
        observation = env.reset()

        while (len(eprew_list) < eval_num):

            for key in observation.keys():
                observation[key] = torch.Tensor(observation[key])

            if render:
                env.render()

            if isinstance(observation, dict):
                goal = observation["desired_goal"]
                observation = observation["observation"]
            else:
                goal = None

            if not self.dicrete_action:
                actions, _, _, _ = self.choose_action(observation, other_data=goal, greedy=True)
            else:
                actions, _ = self.choose_action(observation, other_data=goal, greedy=True)
            actions = actions.cpu().numpy()
            observation, rewards, dones, infos = env.step(actions)
            for e, info in enumerate(infos):
                if dones[e]:
                    eprew_list.append(info.get('episode')['r'] + self.max_steps)
                    if 'is_success' in info.keys():
                        success_history.append(info.get('is_success'))
                    for k in observation.keys():
                        observation[k][e] = info.get('new_obs')[k]

        if len(success_history) > 0:
            return eprew_list, success_history
        else:
            return eprew_list
