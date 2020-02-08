from .PG import PG, PG_Softmax, PG_Gaussian
from .NPG import NPG, NPG_Softmax, NPG_Gaussian
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from .Agent import Agent
from .config import AdaptiveKLPPO_CONFIG,PPO_CONFIG
import copy
import abc
from utils.mathutils import explained_variance
from collections import deque

class PPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self, hyperparams):
        config = copy.deepcopy(PPO_CONFIG)
        config.update(hyperparams)
        super(PPO, self).__init__(config)
        self.batch_size = self.nsteps // config['nbatch_per_iter']
        self.nupdates = config['updates_per_iter']
        self.epsilon = config['clip_epsilon']
        self.v_coef = config["v_coef"]
        self.clip_frac = 0.
        self.beta = 0

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        # update policy
        inds = np.arange(self.nsteps)
        self.clip_frac = 0.
        self.policy_ent = 0.
        self.policy_loss = 0.
        self.value_loss = 0.
        self.old_V = self.V.clone()
        for _ in range(self.nupdates):
            np.random.shuffle(inds)
            for start in range(0, self.nsteps, self.batch_size):
                end = start + self.batch_size
                selected_inds = inds[start:end]
                # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
                imp_fac = self.compute_imp_fac(selected_inds)
                # values and advantages are all 2-D Tensor. size: r.size(0) x 1
                entropy = self.compute_entropy(selected_inds)
                self.policy_ent += entropy.item()
                self.A_batch = self.A[selected_inds]
                self.loss1 = imp_fac * self.A_batch
                self.loss2 = torch.clamp(imp_fac,1.0 - self.epsilon, 1.0 + self.epsilon) * self.A_batch
                clip_frac_batch = (torch.sum(torch.abs(imp_fac - 1) > self.epsilon).float() / self.batch_size).item()
                self.clip_frac += clip_frac_batch
                self.loss = - torch.min(self.loss1, self.loss2).mean() - self.entropy_weight * entropy
                self.policy_loss += self.loss.item()
                if self.value_type is not None:
                    self.update_value(selected_inds)
                self.policy.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
                self.optimizer.step()
                self.v_optimizer.step()
        self.cur_kl = self.mean_kl_divergence().item()
        self.clip_frac /= self.nupdates * (self.nsteps // self.batch_size)
        self.policy_ent /= self.nupdates * (self.nsteps // self.batch_size)
        self.policy_loss /= self.nupdates * (self.nsteps // self.batch_size)
        self.value_loss /= self.nupdates * (self.nsteps // self.batch_size)
        self.learn_step_counter += 1

    def update_value(self, inds = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        V_target = self.esti_R[inds]
        V_eval = self.value(self.s[inds]).squeeze()
        V_eval_clipped = self.old_V[inds] + torch.clamp(V_eval - self.old_V[inds], -self.epsilon, self.epsilon)
        self.loss_v1 = torch.pow(V_eval - V_target, 2)
        self.loss_v2 = torch.pow(V_eval_clipped - V_target, 2)
        self.loss_v = self.v_coef * 0.5 * torch.max(self.loss_v1, self.loss_v2).mean()
        # self.loss_v = self.loss_v1.mean()
        self.value_loss += self.loss_v.item()
        self.value.zero_grad()
        self.loss_v.backward()

class PPO_Gaussian(PPO, NPG_Gaussian):
    def __init__(self,hyperparams):
        super(PPO_Gaussian,self).__init__(hyperparams)

class PPO_Softmax(PPO, NPG_Softmax):
    def __init__(self,hyperparams):
        super(PPO_Softmax,self).__init__(hyperparams)


# adaptive KL PPO is the same with NPG except that KL bound will adaptively increase or decrease after each update
class AdaptiveKLPPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(AdaptiveKLPPO_CONFIG)
        config.update(hyperparams)
        super(AdaptiveKLPPO, self).__init__(config)
        self.nupdates = config['updates_per_iter']
        self.max_grad_norm = config['max_grad_norm']
        self.batch_size = self.nsteps // config['nbatch_per_iter']
        self.beta = config['init_beta']
        self.v_coef = config["v_coef"]
        self.clip_frac = 0. # for printing, no other usage
        self.cur_kl = 0

    def update_beta(self):
        if self.cur_kl < self.max_kl / 1.5:
            self.beta /= 2
        elif self.cur_kl > self.max_kl * 1.5:
            self.beta *= 2

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        # update policy
        inds = np.arange(self.nsteps)
        self.policy_ent = 0.
        self.policy_loss = 0.
        self.value_loss = 0.
        for _ in range(self.nupdates):
            np.random.shuffle(inds)
            for start in range(0, self.nsteps, self.batch_size):
                end = start + self.batch_size
                selected_inds = inds[start:end]
                # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
                imp_fac = self.compute_imp_fac(selected_inds)
                # values and advantages are all 2-D Tensor. size: r.size(0) x 1
                entropy = self.compute_entropy(selected_inds)
                self.policy_ent += entropy.item()
                self.A_batch = self.A[selected_inds]
                # estimate E[kl]
                cur_kl = self.mean_kl_divergence(selected_inds)
                self.loss = - (imp_fac * self.A_batch).mean() + self.beta * cur_kl - self.entropy_weight * entropy
                self.policy_loss += self.loss.item()
                if self.value_type is not None:
                    self.update_value(selected_inds)
                self.policy.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
                self.optimizer.step()
                self.v_optimizer.step()
        # update panishment of kl divergence
        self.cur_kl = self.mean_kl_divergence().item()
        self.update_beta()
        self.policy_ent /= self.nupdates * (self.nsteps // self.batch_size)
        self.policy_loss /= self.nupdates * (self.nsteps // self.batch_size)
        self.value_loss /= self.nupdates * (self.nsteps // self.batch_size)
        self.learn_step_counter += 1

    def update_value(self, inds = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        V_target = self.esti_R[inds]
        V_eval = self.value(self.s[inds]).squeeze()
        self.loss_v = self.v_coef * self.loss_func_v(V_eval, V_target)
        self.value_loss += self.loss_v.item()
        self.value.zero_grad()
        self.loss_v.backward()

class AdaptiveKLPPO_Gaussian(AdaptiveKLPPO, NPG_Gaussian):
    def __init__(self,hyperparams):
        super(AdaptiveKLPPO_Gaussian,self).__init__(hyperparams)

class AdaptiveKLPPO_Softmax(AdaptiveKLPPO, NPG_Softmax):
    def __init__(self,hyperparams):
        super(AdaptiveKLPPO_Softmax,self).__init__(hyperparams)

def run_ppo_train(env, agent, max_timesteps, logger):
    timestep_counter = 0
    total_updates = max_timesteps // agent.nsteps
    epinfobuf = deque(maxlen=100)

    while(True):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_logpacs, mb_obs_, mb_mus, mb_sigmas \
            , mb_distris= [], [], [], [], [], [], [], [], []
        epinfos = []
        observations = env.reset()
        for i in range(0, agent.nsteps, env.num_envs):
            observations = torch.Tensor(observations)
            if not agent.dicrete_action:
                actions, mus, logsigmas, sigmas = agent.choose_action(observations)
                logp = agent.compute_logp(mus, logsigmas, sigmas, actions)
                mus = mus.cpu().numpy()
                sigmas = sigmas.cpu().numpy()
                mb_mus.append(mus)
                mb_sigmas.append(sigmas)
            else:
                actions, distris = agent.choose_action(observations)
                logp = agent.compute_logp(distris, actions)
                distris = distris.cpu().numpy()
                mb_distris.append(distris)
            observations = observations.cpu().numpy()
            actions  = actions.cpu().numpy()
            logp = logp.cpu().numpy()
            observations_, rewards, dones, infos = env.step(actions)

            mb_obs.append(observations)
            mb_actions.append(actions)
            mb_logpacs.append(logp)
            mb_dones.append(dones.astype(np.uint8))
            mb_rewards.append(rewards)
            mb_obs_.append(observations_)

            for e, info in enumerate(infos):
                if dones[e]:
                    epinfos.append(info.get('episode'))
                    observations_[e] = info.get('new_obs')

            observations = observations_

        epinfobuf.extend(epinfos)
        # make all final states marked by done, preventing wrong estimating of returns and advantages.
        # done flag:
        #      0: undone and not the final state
        #      1: realdone
        #      2: undone but the final state
        mb_dones[-1][np.where(mb_dones[-1] == 0)] = 2

        def reshape_data(arr):
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        mb_obs = reshape_data(np.asarray(mb_obs, dtype=np.float32))
        mb_rewards = reshape_data(np.asarray(mb_rewards, dtype=np.float32))
        mb_actions = reshape_data(np.asarray(mb_actions))
        mb_logpacs = reshape_data(np.asarray(mb_logpacs, dtype=np.float32))
        mb_dones = reshape_data(np.asarray(mb_dones, dtype=np.uint8))
        mb_obs_ = reshape_data(np.asarray(mb_obs_, dtype=np.float32))

        assert mb_obs.ndim <= 2 and mb_rewards.ndim <= 2 and mb_actions.ndim <= 2 and \
               mb_logpacs.ndim <= 2 and mb_dones.ndim <= 2 and mb_obs_.ndim <= 2, \
            "databuffer only supports 1-D data's batch."

        if not agent.dicrete_action:
            mb_mus = reshape_data(np.asarray(mb_mus, dtype=np.float32))
            mb_sigmas = reshape_data(np.asarray(mb_sigmas, dtype=np.float32))
            assert mb_mus.ndim <= 2 and mb_sigmas.ndim <= 2, "databuffer only supports 1-D data's batch."
        else:
            mb_distris = reshape_data(np.asarray(mb_distris, dtype=np.float32))
            assert mb_distris.ndim <= 2, "databuffer only supports 1-D data's batch."

        # store transition
        transition = {
            'state': mb_obs if mb_obs.ndim == 2 else np.expand_dims(mb_obs, 1),
            'action': mb_actions if mb_actions.ndim == 2 else np.expand_dims(mb_actions, 1),
            'reward': mb_rewards if mb_rewards.ndim == 2 else np.expand_dims(mb_rewards, 1),
            'next_state': mb_obs_ if mb_obs_.ndim == 2 else np.expand_dims(mb_obs_, 1),
            'done': mb_dones if mb_dones.ndim == 2 else np.expand_dims(mb_dones, 1),
            'logpac': mb_logpacs if mb_logpacs.ndim == 2 else np.expand_dims(mb_logpacs, 1),
        }
        if not agent.dicrete_action:
            transition['mu'] = mb_mus if mb_mus.ndim == 2 else np.expand_dims(mb_mus, 1)
            transition['sigma'] = mb_sigmas if mb_sigmas.ndim == 2 else np.expand_dims(mb_sigmas, 1)
        else:
            transition['distri'] = mb_distris if mb_distris.ndim == 2 else np.expand_dims(mb_distris, 1)
        agent.store_transition(transition)

        # agent learning step
        agent.learn()

        # training controller
        timestep_counter += agent.nsteps
        if timestep_counter >= max_timesteps:
            break

        # adjust learning rate for policy and value function
        decay_coef = 1 - agent.learn_step_counter / total_updates
        adjust_learning_rate(agent.optimizer, original_lr=agent.lr, decay_coef=decay_coef)
        if agent.value_type is not None:
            adjust_learning_rate(agent.v_optimizer, original_lr=agent.lr_v, decay_coef=decay_coef)

        print("------------------log information------------------")
        print("total_timesteps:".ljust(20) + str(timestep_counter))
        print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
        explained_var = explained_variance(agent.V.cpu().numpy(), agent.esti_R.cpu().numpy())
        print("explained_var:".ljust(20) + str(explained_var))
        logger.add_scalar("explained_var/train", explained_var, timestep_counter)
        print("episode_len:".ljust(20) + "{:.1f}".format(np.mean([epinfo['l'] for epinfo in epinfobuf])))
        print("episode_rew:".ljust(20) + str(np.mean([epinfo['r'] for epinfo in epinfobuf])))
        logger.add_scalar("episode_reward/train", np.mean([epinfo['r'] for epinfo in epinfobuf]), timestep_counter)
        print("mean_kl:".ljust(20) + str(agent.cur_kl))
        logger.add_scalar("mean_kl/train", agent.cur_kl, timestep_counter)
        print("policy_ent:".ljust(20) + str(agent.policy_ent))
        logger.add_scalar("policy_ent/train", agent.policy_ent, timestep_counter)
        print("policy_loss:".ljust(20)+ str(agent.policy_loss))
        logger.add_scalar("policy_loss/train", agent.policy_loss, timestep_counter)
        print("value_loss:".ljust(20)+ str(agent.value_loss))
        logger.add_scalar("value_loss/train", agent.value_loss, timestep_counter)
        print("clip_frac:".ljust(20) + "{:.4f}".format(agent.clip_frac) + "(only for standard PPO)")
        print("kl_panishment:".ljust(20) + "{:.4f}".format(agent.beta) + "(only for Adaptive KL PPO)")
    return agent

def adjust_learning_rate(optimizer, original_lr = 1e-4, decay_coef = 0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr * decay_coef
