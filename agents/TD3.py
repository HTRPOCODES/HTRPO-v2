import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.Agent import Agent
from .config import DDPG_CONFIG
import basenets
import copy
from utils import databuffer
import os
from collections import deque
from utils.mathutils import explained_variance
from .DDPG import DDPG
from rlnets.DDPG import FCDDPG_C
from .config import TD3_CONFIG

class TD3(DDPG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(TD3_CONFIG)
        config.update(hyperparams)
        super(TD3, self).__init__(config)

        self.d = config["actor_delayed_steps"]
        self.smooth_noise = config["smooth_noise"]
        self.epsilon = config["smooth_epsilon"]

        self.e_Critic_double = FCDDPG_C(self.n_states, self.n_action_dims,
                                 n_hiddens=self.hidden_layers_v,
                                 nonlinear=self.act_func,
                                 usebn=self.using_bn,
                                 initializer="uniform",
                                 initializer_param={"last_lower": 3e-3, "last_upper": 3e-3}
                                 )
        self.t_Critic_double = FCDDPG_C(self.n_states, self.n_action_dims,
                                 n_hiddens=self.hidden_layers_v,
                                 nonlinear=self.act_func,
                                 usebn=self.using_bn,
                                 initializer="uniform",
                                 initializer_param={"last_lower": 3e-3, "last_upper": 3e-3}
                                 )
        self.hard_update(self.t_Critic, self.e_Critic)
        self.optimizer_c = self.optimizer_c_func(list(self.e_Critic.parameters()) +
                                                 list(self.e_Critic_double.parameters()),
                                                 lr = self.lrv)

    def cuda(self):
        DDPG.cuda(self)
        self.e_Critic_double.cuda()
        self.t_Critic_double.cuda()

    def learn(self):

        for i in range(self.d):
            # sample batch memory from all memory
            batch_memory = self.sample_batch(self.batch_size)[0]
            if self.norm_ob:
                batch_memory['state'] = np.clip(
                    (batch_memory['state'] - self.ob_mean) / np.sqrt(self.ob_var + 1e-8), -10, 10)
                batch_memory['next_state'] = np.clip(
                    (batch_memory['next_state'] - self.ob_mean) / np.sqrt(self.ob_var + 1e-8), -10, 10)
            if self.norm_rw:
                batch_memory['reward'] = np.clip(batch_memory['reward'] / np.sqrt(self.rw_var + 1e-8), -10, 10)
            self.r = self.r.resize_(batch_memory['reward'].shape).copy_(torch.Tensor(batch_memory['reward']))
            self.done = self.done.resize_(batch_memory['done'].shape).copy_(torch.Tensor(batch_memory['done']))
            self.s_ = self.s_.resize_(batch_memory['next_state'].shape).copy_(torch.Tensor(batch_memory['next_state']))
            self.a = self.a.resize_(batch_memory['action'].shape).copy_(torch.Tensor(batch_memory['action']))
            self.s = self.s.resize_(batch_memory['state'].shape).copy_(torch.Tensor(batch_memory['state']))

            # Target Policy Smoothing'
            a_noise = np.clip(np.random.normal(0, self.smooth_noise, size = self.a.size()), -self.epsilon, self.epsilon)
            a_noise = torch.Tensor(a_noise).type_as(self.a)
            a_ = torch.clamp(self.t_Actor(self.s_) + a_noise, -self.action_bounds, self.action_bounds)

            # Clipping Double Q Learning
            q_1 = self.t_Critic(self.s_, a_)
            q_2 = self.t_Critic_double(self.s_, a_)
            q_target = self.r + (1 - self.done) * self.gamma * torch.min(q_1, q_2)
            q_target = q_target.detach().squeeze()

            q_eval_1 = self.e_Critic(self.s, self.a).squeeze()
            q_eval_2 = self.e_Critic_double(self.s, self.a).squeeze()
            self.Qt = q_target.cpu().numpy()
            self.Qe1 = q_eval_1.detach().cpu().numpy()
            self.Qe2 = q_eval_2.detach().cpu().numpy()

            # update critic
            self.loss_c = self.loss(q_eval_1, q_target) + self.loss(q_eval_2, q_target)

            self.e_Critic.zero_grad()
            self.e_Critic_double.zero_grad()
            self.loss_c.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.e_Critic.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.e_Critic_double.parameters(), self.max_grad_norm)
            self.optimizer_c.step()

        # update actor
        self.loss_a = -self.e_Critic(self.s, self.e_Actor(self.s)).mean()
        self.e_Actor.zero_grad()
        self.loss_a.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.e_Actor.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (
                1 - self.exploration_noise_decrement) if self.noise > self.noise_min else self.noise_min

        # check to replace target parameters
        self.soft_update(self.t_Actor, self.e_Actor, self.replace_tau)
        self.soft_update(self.t_Critic, self.e_Critic, self.replace_tau)
        self.soft_update(self.t_Critic_double, self.e_Critic_double, self.replace_tau)

def run_td3_train(env, agent, max_timesteps, logger, log_interval):

    timestep_counter = 0
    total_updates = max_timesteps / env.num_envs
    epinfobuf = deque(maxlen=100)
    observations = env.reset()

    loss_a = 0
    loss_c = 0
    explained_var = 0

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
                # adjust_learning_rate(agent.optimizer_a, original_lr=agent.lr, decay_coef=decay_coef)
                # adjust_learning_rate(agent.optimizer_c, original_lr=agent.lrv, decay_coef=decay_coef)

                explained_var += 0.5 * explained_variance(agent.Qe1, agent.Qt)
                explained_var += 0.5 * explained_variance(agent.Qe2, agent.Qt)
                loss_a += agent.loss_a.item()
                loss_c += agent.loss_c.item()
                if agent.learn_step_counter % log_interval == 0:
                    print("------------------log information------------------")
                    print("total_timesteps:".ljust(20) + str(timestep_counter))
                    print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
                    print("explained_var:".ljust(20) + str(explained_var / log_interval))
                    logger.add_scalar("explained_var/train", explained_var / log_interval, timestep_counter)
                    print("episode_len:".ljust(20) + "{:.1f}".format(np.mean([epinfo['l'] for epinfo in epinfobuf])))
                    print("episode_rew:".ljust(20) + str(np.mean([epinfo['r'] for epinfo in epinfobuf])))
                    logger.add_scalar("episode_reward/train", np.mean([epinfo['r'] for epinfo in epinfobuf]),
                                      timestep_counter)
                    print("max_episode_rew:".ljust(20) + str(np.max([epinfo['r'] for epinfo in epinfobuf])))
                    print("min_episode_rew:".ljust(20) + str(np.min([epinfo['r'] for epinfo in epinfobuf])))
                    print("loss_a:".ljust(20) + str(loss_a / log_interval))
                    logger.add_scalar("actor_loss/train", loss_a / log_interval, timestep_counter)
                    print("loss_c:".ljust(20) + str(loss_c / log_interval))
                    logger.add_scalar("critic_loss/train", loss_c / log_interval, timestep_counter)
                    print("action_noise_std:".ljust(20) + str(agent.noise))

                    explained_var = 0
                    loss_a = 0
                    loss_c = 0

    return agent

def adjust_learning_rate(optimizer, original_lr=1e-4, decay_coef=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr * decay_coef

