"""
baseline A2C for AntDirection
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


parser = argparse.ArgumentParser(description='baseline A2C for AntDirection')
parser.add_argument('--env_name', default="AntDirection-v1", help='environment name')
parser.add_argument('--adapt_lr', type=float, default=0.1,help='adapt lr (default: 0.1)')
parser.add_argument('--tau', type=float, default=1.0, help='tau (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma (default: 0.95)')
parser.add_argument('--adapt_steps', type=int, default=1, help='adapt steps (default: 1)')
parser.add_argument('--meta_bsz', type=int, default=20, help='meta_bsz (default: 20)')
parser.add_argument('--adapt_bsz', type=int, default=20, help='adapt_bsz (default: 20)')
parser.add_argument('--seed', type=int, default=42, help='seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=10, help='num cores (default: 10)')
parser.add_argument('--cuda', type=int, default=1, help='cuda index (default: 1, 0 : cpu)')
args = parser.parse_args()

def main():
    env_name = args.env_name
    adapt_lr = args.adapt_lr
    tau = args.tau
    gamma = args.gamma
    adapt_steps = args.adapt_steps
    meta_bsz = args.meta_bsz
    adapt_bsz = args.adapt_bsz
    seed = args.seed
    num_workers = args.num_workers
    cuda = args.cuda
    
    log_adapt_r = []
    time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("logs/args_antdir_a2c_" + time_start + ".txt", "w") as f:
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
    adaptation_reward = iteration_reward / meta_bsz
    print('adaptation_reward', adaptation_reward)
    log_adapt_r.append(adaptation_reward)
        
    # log results
    log_ar = np.array(log_adapt_r)
    torch.save(policy.state_dict(), "logs/model_antdir_a2c_" + time_start + ".pt")
    np.save("logs/log_antdir_a2c_" + time_start, log_ar)

if __name__ == '__main__':
    main()
