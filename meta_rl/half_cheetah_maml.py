"""
HalfCheetah for MAML
"""

import random
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy

import argparse
import pickle
from datetime import datetime
import json

def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='Surrogate Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl

parser = argparse.ArgumentParser(description='HalfCheetah for MAML')
parser.add_argument('--env_name', default="HalfCheetahForwardBackward-v1", help='environment name')
parser.add_argument('--adapt_lr', type=float, default=0.1,help='adapt lr (default: 0.1)')
parser.add_argument('--meta_lr', type=float, default=1.0, help='meta lr (default: 1.0)')
parser.add_argument('--tau', type=float, default=1.0, help='tau (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma (default: 0.95)')
parser.add_argument('--adapt_steps', type=int, default=1, help='adapt steps (default: 1)')
parser.add_argument('--num_iterations', type=int, default=300, help='num iterations (default: 300)')
parser.add_argument('--meta_bsz', type=int, default=20, help='meta_bsz (default: 20)')
parser.add_argument('--adapt_bsz', type=int, default=20, help='adapt_bsz (default: 20)')
parser.add_argument('--seed', type=int, default=42, help='seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=10, help='num cores (default: 10)')
parser.add_argument('--cuda', type=int, default=1, help='cuda index (default: 1, 0 : cpu)')
parser.add_argument('--log_int', type=int, default=10, help='log interval (default: 10 iterations)')
args = parser.parse_args()

def main():
    env_name = args.env_name
    adapt_lr = args.adapt_lr
    meta_lr = args.meta_lr
    tau = args.tau
    gamma = args.gamma
    adapt_steps = args.adapt_steps
    num_iterations = args.num_iterations
    meta_bsz = args.meta_bsz
    adapt_bsz = args.adapt_bsz
    seed = args.seed
    num_workers = args.num_workers
    cuda = args.cuda
    log_int = args.log_int
    
    log_adapt_r = []
    time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("logs/args_half_cheetah_maml" + time_start + ".txt", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cpu'
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device_name = 'cuda'
        print('using CUDA')
    device = torch.device(device_name)

    def make_env():
        env = gym.make(env_name)
        env = ch.envs.ActionSpaceScaler(env)
        return env

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    policy = DiagNormalPolicy(env.state_size, env.action_size, device=device)
    if cuda:
        policy = policy.to(device)
    baseline = LinearValue(env.state_size, env.action_size)

    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(clone, episodes=adapt_bsz)
                if cuda:
                    train_episodes = train_episodes.to(device, non_blocking=True)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)
        log_adapt_r.append(adaptation_reward)
	
        # TRPO meta-optimization
        backtrack_factor = 0.5
        ls_max_steps = 15
        max_kl = 0.01
        if cuda:
            policy = policy.to(device, non_blocking=True)
            baseline = baseline.to(device, non_blocking=True)
            iteration_replays = [[r.to(device, non_blocking=True) for r in task_replays] for task_replays in
                                 iteration_replays]

        # Compute CG step direction
        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma,
                                               adapt_lr)
        grad = autograd.grad(old_loss,
                             policy.parameters(),
                             retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line-search
        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor ** ls_step * meta_lr
            clone = deepcopy(policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(-stepsize, u.data)
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, tau, gamma,
                                               adapt_lr)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                break
        
        # log results
        if iteration % log_int == 0:
            log_ar = np.array(log_adapt_r)
            torch.save(policy.state_dict(), "logs/model_half_cheetah_maml" + time_start + ".pt")
            np.save("logs/log_half_cheetah_maml" + time_start, log_ar)

if __name__ == '__main__':
    main()
