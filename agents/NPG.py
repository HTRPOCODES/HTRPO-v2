from agents.PG import PG, PG_Gaussian, PG_Softmax
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from .config import NPG_CONFIG
import abc
from utils.mathutils import explained_variance
from collections import deque

class NPG(PG):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(NPG_CONFIG)
        config.update(hyperparams)
        super(NPG, self).__init__(config)
        self.cg_iters = config['cg_iters']
        self.cg_residual_tol = config['cg_residual_tol']
        self.cg_damping = config['cg_damping']
        self.max_kl = config['max_kl_divergence']

    def conjunction_gradient(self, b):
        """
        Demmel p 312, borrowed from https://github.com/ikostrikov/pytorch-trpo
        """
        p = b.clone().data
        r = b.clone().data
        x = torch.zeros_like(b).data
        rdotr = torch.sum(r * r)
        for i in range(self.cg_iters):
            z = self.hessian_vector_product(Variable(p))
            v = rdotr / torch.sum(p * z.data)
            x += v * p
            r -= v * z.data
            newrdotr = torch.sum(r * r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < self.cg_residual_tol:
                break
        return Variable(x)

    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        Borrowed from https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py
        """
        self.policy.zero_grad()
        mean_kl_div = self.mean_kl_divergence()
        kl_grad = torch.autograd.grad(
            mean_kl_div, self.policy.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        grad_grad = torch.autograd.grad(
            grad_vector_product, self.policy.parameters())
        fisher_vector_product = torch.cat(
            [grad.contiguous().view(-1) for grad in grad_grad])
        return fisher_vector_product + (self.cg_damping * vector)

    def learn(self):
        self.sample_batch()
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac()
        # values and advantages are all 2-D Tensor. size: r.size(0) x 1
        self.estimate_value()
        # update policy
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        self.loss = - (imp_fac * self.A.squeeze()).mean()
        # update value
        if self.value_type is not None:
            for i in range(self.iters_v):
                self.update_value()
        self.policy.zero_grad()
        loss_grad = torch.autograd.grad(
            self.loss, self.policy.parameters(), create_graph=True)
        # loss_grad_vector is a 1-D Variable including all parameters in self.policy
        loss_grad_vector = parameters_to_vector([grad for grad in loss_grad])
        # solve Ax = -g, A is Hessian Matrix of KL divergence
        trpo_grad_direc = self.conjunction_gradient(- loss_grad_vector)
        shs = .5 * torch.sum(trpo_grad_direc * self.hessian_vector_product(trpo_grad_direc))
        beta = torch.sqrt(self.max_kl / shs)
        fullstep = trpo_grad_direc * beta
        thetanew = parameters_to_vector(self.policy.parameters()) + fullstep
        vector_to_parameters(thetanew, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_ent = self.compute_entropy().item()


class NPG_Gaussian(NPG, PG_Gaussian):
    def __init__(self,hyperparams):
        super(NPG_Gaussian, self).__init__(hyperparams)

class NPG_Softmax(NPG,PG_Softmax):
    def __init__(self,hyperparams):
        super(NPG_Softmax, self).__init__(hyperparams)

def run_npg_train(env, agent, max_timesteps, logger):
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
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_obs.append(observations)
            mb_actions.append(actions)
            mb_logpacs.append(logp)
            mb_dones.append(dones.astype(np.uint8))
            mb_rewards.append(rewards)
            mb_obs_.append(observations_)
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

        print("------------------log information------------------")
        print("total_timesteps:".ljust(20) + str(timestep_counter))
        print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
        if agent.value_type is not None:
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
        print("value_loss:".ljust(20)+ str(agent.value_loss))
        logger.add_scalar("value_loss/train", agent.value_loss, timestep_counter)
    return agent



