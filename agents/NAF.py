import torch
import numpy as np
import copy
from agents.Agent import Agent
from .config import NAF_CONFIG
from rlnets.NAF import FCNAF
from utils import databuffer
import os
from collections import deque
from utils.mathutils import explained_variance

class NAF(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(NAF_CONFIG)
        config.update(hyperparams)
        super(NAF, self).__init__(config)
        # NAF configs
        self.replace_tau = config['tau']
        self.noise = np.sqrt(config['noise_var'])
        self.noise_min = config['noise_min']
        self.exploration_noise_decrement = config['noise_decrease']
        self.loss_func = config['loss']()
        self.batch_size = config['batch_size']
        self.action_bounds = config['action_bounds']
        self.max_grad_norm = config['max_grad_norm']
        self.nsteps = config['steps_per_iter']
        self.learn_start_step = config['learn_start_step']
        self.norm_ob = config['norm_ob']
        if self.norm_ob:
            self.ob_mean = 0
            self.ob_var = 1
        self.norm_rw = config['norm_rw']
        if self.norm_rw:
            self.rw_mean = 0
            self.rw_var = 1
        # initialize zero memory [s, a, r, s_]
        if "memory_size" not in config.keys():
            config["memory_size"] = self.memory_size
        self.memory = databuffer(config)
        self.e_NAF = FCNAF(self.n_states, self.n_action_dims,
                           n_hiddens=self.hidden_layers,
                           usebn=self.using_bn,
                           nonlinear=self.act_func,
                           action_active=self.out_act_func,
                           action_scaler=self.action_bounds)
        self.t_NAF = FCNAF(self.n_states, self.n_action_dims,
                           n_hiddens=self.hidden_layers,
                           usebn=self.using_bn,
                           nonlinear=self.act_func,
                           action_active=self.out_act_func,
                           action_scaler=self.action_bounds)
        self.optimizer = config['optimizer'](self.e_NAF.parameters(), lr=self.lr)
        self.hard_update(self.t_NAF, self.e_NAF)

    def cuda(self):
        Agent.cuda(self)
        self.e_NAF = self.e_NAF.cuda()
        self.t_NAF = self.t_NAF.cuda()

    def choose_action(self,s):
        if self.norm_ob:
            s = torch.clamp(
                (s - torch.Tensor(self.ob_mean).type_as(s)) / torch.sqrt(torch.Tensor(self.ob_var).type_as(s) + 1e-8),
                -10,10)
        self.e_NAF.eval()
        s = torch.Tensor(s)
        if self.use_cuda:
            s = s.cuda()
        _, preda,_ = self.e_NAF(s)
        self.e_NAF.train()
        anoise = torch.normal(torch.zeros(preda.size()),
                              self.noise * torch.ones(preda.size())).type_as(preda)
        return (preda + anoise).detach()

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_NAF, self.e_NAF, self.replace_tau)

        # sample batch memory from all memory
        batch_memory = self.sample_batch(self.batch_size)[0]
        if self.norm_ob:
            batch_memory['state'] = np.clip(
                (batch_memory['state'] - self.ob_mean) / np.sqrt(self.ob_var + 1e-8),-10,10)
            batch_memory['next_state'] = np.clip(
                (batch_memory['next_state'] - self.ob_mean) / np.sqrt(self.ob_var + 1e-8),-10,10)
        if self.norm_rw:
            batch_memory['reward'] = np.clip(batch_memory['reward'] / np.sqrt(self.rw_var + 1e-8), -10, 10)
        self.r = self.r.resize_(batch_memory['reward'].shape).copy_(torch.Tensor(batch_memory['reward']))
        self.done = self.done.resize_(batch_memory['done'].shape).copy_(torch.Tensor(batch_memory['done']))
        self.s_ = self.s_.resize_(batch_memory['next_state'].shape).copy_(torch.Tensor(batch_memory['next_state']))
        self.a = self.a.resize_(batch_memory['action'].shape).copy_(torch.Tensor(batch_memory['action']))
        self.s = self.s.resize_(batch_memory['state'].shape).copy_(torch.Tensor(batch_memory['state']))

        V_, _, _ = self.t_NAF(self.s_)
        q_target = self.r + self.gamma * V_
        q_target = q_target.squeeze().detach()
        self.Qt = q_target.cpu().numpy()

        V,mu,L = self.e_NAF(self.s)
        a_mu = self.a - mu
        a_muxL = torch.bmm(a_mu.unsqueeze(1),L)
        A = -0.5 * torch.bmm( a_muxL, a_muxL.transpose(1,2))
        q_eval = V.squeeze() + A.squeeze()
        self.Qe = q_eval.detach().cpu().numpy()

        self.e_NAF.zero_grad()
        self.loss = self.loss_func(q_eval, q_target)
        self.loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.e_NAF.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (1 - self.exploration_noise_decrement) \
                     if self.noise > self.noise_min else self.noise_min

    def save_model(self, save_path):
        print("saving models...")
        save_dict = {
            'model': self.e_NAF.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'noise': self.noise,
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
        }
        torch.save(save_dict, os.path.join(save_path, "policy" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        policy_name = os.path.join(load_path, "policy" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (policy_name))
        checkpoint = torch.load(policy_name)
        self.e_NAF.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.noise = checkpoint['noise']
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        print("loaded checkpoint %s" % (policy_name))

def run_naf_train(env, agent, max_timesteps, logger, log_interval):
    timestep_counter = 0
    total_updates = max_timesteps / env.num_envs
    epinfobuf = deque(maxlen=100)
    observations = env.reset()
    curlen = np.zeros(shape=(observations.shape[0], 1))

    while (True):

        # collection of training data
        mb_obs, mb_as, mb_dones, mb_rs, mb_obs_ = [], [], [], [], []
        epinfos = []
        for i in range(0, agent.nsteps, env.num_envs):
            observations = torch.Tensor(observations)
            if timestep_counter > agent.learn_start_step:
                actions = agent.choose_action(observations)
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            else:
                actions = []
                for i in range(env.num_envs):
                    actions.append(env.action_space.sample())
                actions = np.asarray(actions, dtype=np.float32)

            observations = observations.cpu().numpy()
            observations_, rewards, dones, infos = env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            mb_obs.append(observations)
            mb_as.append(actions)
            mb_rs.append(rewards)
            mb_obs_.append(observations_)
            mb_dones.append(dones)

            observations = observations_

        epinfobuf.extend(epinfos)

        def reshape_data(arr):
            s = arr.shape
            return arr.reshape(s[0] * s[1], *s[2:])
        mb_obs = reshape_data(np.asarray(mb_obs, dtype=np.float32))
        mb_rs = reshape_data(np.asarray(mb_rs, dtype=np.float32))
        mb_as = reshape_data(np.asarray(mb_as))
        mb_dones = reshape_data(np.asarray(mb_dones, dtype=np.uint8))
        mb_obs_ = reshape_data(np.asarray(mb_obs_, dtype=np.float32))

        # store transition
        transition = {
            'state': mb_obs if mb_obs.ndim == 2 else np.expand_dims(mb_obs, 1),
            'action': mb_as if mb_as.ndim == 2 else np.expand_dims(mb_as, 1),
            'reward': mb_rs if mb_rs.ndim == 2 else np.expand_dims(mb_rs, 1),
            'next_state': mb_obs_ if mb_obs_.ndim == 2 else np.expand_dims(mb_obs_, 1),
            'done': mb_dones if mb_dones.ndim == 2 else np.expand_dims(mb_dones, 1),
        }
        agent.store_transition(transition)

        # training controller
        timestep_counter += agent.nsteps
        if timestep_counter >= max_timesteps:
            break

        if timestep_counter > agent.batch_size:
            # Update observation and reward mean and var.
            if agent.norm_ob:
                agent.ob_mean, agent.ob_var = env.ob_rms.mean, env.ob_rms.var
            if agent.norm_rw:
                agent.rw_mean, agent.rw_var = env.ret_rms.mean, env.ret_rms.var
            for i in range(0, agent.nsteps):
                agent.learn()
            # adjust learning rate for policy and value function
            # decay_coef = 1 - agent.learn_step_counter / total_updates
            # adjust_learning_rate(agent.optimizer, original_lr=agent.lr, decay_coef=decay_coef)
            if agent.learn_step_counter % log_interval == 0:
                print("------------------log information------------------")
                print("total_timesteps:".ljust(20) + str(timestep_counter))
                print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
                explained_var = explained_variance(agent.Qe, agent.Qt)
                print("explained_var:".ljust(20) + str(explained_var))
                logger.add_scalar("explained_var/train", explained_var, timestep_counter)
                print("episode_len:".ljust(20) + "{:.1f}".format(np.mean([epinfo['l'] for epinfo in epinfobuf])))
                print("episode_rew:".ljust(20) + str(np.mean([epinfo['r'] for epinfo in epinfobuf])))
                logger.add_scalar("episode_reward/train", np.mean([epinfo['r'] for epinfo in epinfobuf]),
                                  timestep_counter)
                print("max_episode_rew:".ljust(20) + str(np.max([epinfo['r'] for epinfo in epinfobuf])))
                print("min_episode_rew:".ljust(20) + str(np.min([epinfo['r'] for epinfo in epinfobuf])))
                print("loss:".ljust(20) + str(agent.loss.item()))
                logger.add_scalar("value_loss/train", agent.loss.item(), timestep_counter)

    return agent

def adjust_learning_rate(optimizer, original_lr = 1e-4, decay_coef = 0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr * decay_coef
