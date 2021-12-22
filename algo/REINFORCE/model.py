import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ExpReplay
from collections import namedtuple
from torch.distributions import Categorical
from utils.Algorithm import Algorithm


class REINFORCE(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate):
        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.policy  = Net(self.dim_in, self.dim_out)
        self.gamma   = disc_rate
        self.transition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'dones'))
        self.buffer = ExpReplay(10000, self.transition)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def reset(self):
        self.buffer.clear()

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  # change to tensor
        prob = self.policy.forward(x)
        dist = Categorical(prob)

        action = dist.sample()  # action sampled from defined distribution
        log_prob = dist.log_prob(action)  # log_prob of pi(a|s)
        return action.item(), log_prob

    def store(self, *args):
        self.buffer.store(*args)

    def train(self):
        transitions = self.buffer.sample(self.buffer.__len__())
        batch = self.transition(*zip(*transitions))
        reward_list = batch.reward

        # calculate return of all times in the episode
        T = len(reward_list)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        # calculate returns recursively
        for t in reversed(range(T)):
            future_return = reward_list[t] + self.gamma * future_return
            returns[t] = future_return

        # calculate loss of policy
        # loss of REINFORCE algorithm is
        # the expectation of Return, G * grad of log pi(a|s) over a distribution pi_theta.
        returns = torch.tensor(returns)
        log_probs = torch.stack(batch.logprobs)
        loss = - log_probs * returns
        loss = torch.sum(loss)

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()