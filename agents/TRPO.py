from .NPG import NPG, NPG_Gaussian, NPG_Softmax
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from .config import TRPO_CONFIG
import abc
import numpy as np
from utils.mathutils import explained_variance
from collections import deque
import time

class TRPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self, hyperparams):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(hyperparams)
        super(TRPO, self).__init__(config)
        self.accept_ratio = config['accept_ratio']
        self.max_search_num = config['max_search_num']
        self.step_frac = config['step_frac']
        self.improvement = 0
        self.expected_improvement = 0

    def object_loss(self, theta):
        model = copy.deepcopy(self.policy)
        vector_to_parameters(theta, model.parameters())
        imp_fac = self.compute_imp_fac(model=model)
        loss = - (imp_fac * self.A).mean() - self.entropy_weight * self.compute_entropy()
        curkl = self.mean_kl_divergence(model=model)
        return loss, curkl

    def linear_search(self,x, fullstep, expected_improve_rate):
        accept_ratio = self.accept_ratio
        max_backtracks = self.max_search_num
        fval, curkl = self.object_loss(x)
        print("*****************************************")
        for (_n_backtracks, stepfrac) in enumerate(list(self.step_frac ** torch.arange(0, max_backtracks).float().type_as(self.s))):
            xnew = x + stepfrac * fullstep
            newfval, curkl = self.object_loss(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print("Search number {}...".format(_n_backtracks + 1),  "Step Frac:{:.5f}".format(stepfrac),
                  " Actual improve: {:.5f}".format(actual_improve) , " Expected improve: {:.5f}".format(expected_improve),
                  " Current KL: {:.8f}".format(curkl))
            if ratio.item() > accept_ratio and actual_improve.item() > 0 and 0 < curkl < self.max_kl * 1.5:
                self.improvement = actual_improve.item()
                self.expected_improvement = expected_improve.item()
                print("*****************************************")
                return xnew
        print("** Failure optimization, rolling back. **")
        print("*****************************************")
        return x

    def learn(self):
        self.sample_batch()
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac()
        self.estimate_value()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        self.loss = - (imp_fac * self.A).mean() - self.entropy_weight * self.compute_entropy()
        if self.value_type is not None:
            # update value
            for i in range(self.iters_v):
                self.update_value()
        self.policy.zero_grad()
        loss_grad = torch.autograd.grad(
            self.loss, self.policy.parameters(), create_graph=True)
        # loss_grad_vector is a 1-D Variable including all parameters in self.policy
        loss_grad_vector = parameters_to_vector([grad for grad in loss_grad])
        # solve Ax = -g, A is Hessian Matrix of KL divergence
        trpo_grad_direc = self.conjunction_gradient( - loss_grad_vector)
        shs = .5 * torch.sum(trpo_grad_direc * self.hessian_vector_product(trpo_grad_direc))
        beta = torch.sqrt(self.max_kl / shs)
        fullstep = trpo_grad_direc * beta
        gdotstepdir = -torch.sum(loss_grad_vector * trpo_grad_direc)
        theta = self.linear_search(parameters_to_vector(
            self.policy.parameters()), fullstep, gdotstepdir * beta)
        # update policy
        vector_to_parameters(theta, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_ent = self.compute_entropy().item()

class TRPO_Gaussian(TRPO, NPG_Gaussian):
    def __init__(self,hyperparams):
        super(TRPO_Gaussian, self).__init__(hyperparams)

class TRPO_Softmax(TRPO, NPG_Softmax):
    def __init__(self,hyperparams):
        super(TRPO_Softmax, self).__init__(hyperparams)

def run_trpo_train(env, agent, max_timesteps, logger):
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
        print("actual_imprv:".ljust(20) + "{:.3f}".format(agent.improvement))
        logger.add_scalar("actual_imprv/train", agent.improvement, timestep_counter)
        print("exp_imprv:".ljust(20) + "{:.3f}".format(agent.expected_improvement))
        logger.add_scalar("exp_imprv/train", agent.expected_improvement, timestep_counter)

    return agent


