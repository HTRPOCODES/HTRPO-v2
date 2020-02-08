import torch
import numpy as np
import basenets
from agents.Agent import Agent
import copy
from .config import DQN_CONFIG
from rlnets.DQN import FCDQN
from utils import databuffer
import os

class DQN(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        super(DQN,self).__init__(config)
        self.epsilon_max = config['e_greedy']
        self.replace_target_iter = config['replace_target_iter']
        self.epsilon_increment = config['e_greedy_increment']
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        # initialize zero memory [s, a, r, s_]
        config['memory_size'] = self.memory_size
        self.memory = databuffer(config)
        self.batch_size = config['batch_size']
        ## TODO: include other network architectures
        if type(self) == DQN:
            self.e_DQN = FCDQN(self.n_states, self.n_actions,
                               n_hiddens=config['hidden_layers'],
                               usebn=config['use_batch_norm'],
                               nonlinear=config['act_func'])
            self.t_DQN = FCDQN(self.n_states, self.n_actions,
                               n_hiddens=config['hidden_layers'],
                               usebn=config['use_batch_norm'],
                               nonlinear=config['act_func'])
            self.lossfunc = config['loss']()
            if self.mom == 0 or self.mom is None:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(),lr = self.lr)
            else:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(), lr=self.lr, momentum = self.mom)

    def cuda(self):
        Agent.cuda(self)
        self.e_DQN = self.e_DQN.cuda()
        self.t_DQN = self.t_DQN.cuda()

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        observation = torch.Tensor(observation)
        if self.use_cuda:
            observation = observation.cuda()
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            self.e_DQN.eval()
            actions_value = self.e_DQN(observation)
            self.e_DQN.train()
            (_ , action) = torch.max(actions_value, 1)
            distri = actions_value.detach().cpu().numpy()
            action = action[0].cpu().numpy()
        else:
            distri = 1. / self.n_actions * np.ones(self.n_actions)
            action = np.random.randint(0, self.n_actions)
        return action, distri

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.hard_update(self.t_DQN, self.e_DQN)
        batch_memory = self.sample_batch(self.batch_size)[0]

        self.r = self.r.resize_(batch_memory['reward'].shape).copy_(torch.Tensor(batch_memory['reward']))
        self.done = self.done.resize_(batch_memory['done'].shape).copy_(torch.Tensor(batch_memory['done']))
        self.s_ = self.s_.resize_(batch_memory['next_state'].shape).copy_(torch.Tensor(batch_memory['next_state']))
        self.a = self.a.resize_(batch_memory['action'].shape).copy_(torch.Tensor(batch_memory['action']))
        self.s = self.s.resize_(batch_memory['state'].shape).copy_(torch.Tensor(batch_memory['state']))

        q_target = self.r + self.gamma * torch.max(self.t_DQN(self.s_), 1)[0].view(self.batch_size, 1)
        q_eval = self.e_DQN(self.s)
        q_eval_wrt_a = q_eval.gather(1, self.a.long())
        q_target = q_target.detach()

        self.loss = self.lossfunc(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.cost_his.append(self.loss.data)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self, save_path):
        print("saving models...")
        save_dict = {
            'model': self.e_DQN.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
            'epsilon': self.epsilon,
        }
        torch.save(save_dict, os.path.join(save_path, "policy" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        policy_name = os.path.join(load_path, "policy" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (policy_name))
        checkpoint = torch.load(policy_name)
        self.e_DQN.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        self.epsilon = checkpoint['epsilon']
        print("loaded checkpoint %s" % (policy_name))
