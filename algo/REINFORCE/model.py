import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ExpReplay

from torch.distributions import Categorical

gamma = 0.99  # discount rate


class REINFORCE(nn.Module):
    def __init__(self, env, Net, learning_rate, disc_rate):
        super(REINFORCE, self).__init__()

        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.policy  = Net(self.dim_in, self.dim_out)

        self.gamma  = disc_rate
        self.buffer = ExpReplay()
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

    def store(self, state_tuple):
        """
        stores the output of environment and agents into agents experience buffer.

        :param state_tuple:
        :return:
        """
        self.buffer.store(state_tuple)

    def train(self):

        # calculate return of all times in the episode
        reward_list = self.buffer.rewards
        T = len(reward_list)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        # calculate returns recursively
        for t in reversed(range(T)):
            future_return = reward_list[t] + gamma * future_return
            returns[t] = future_return

        # calculate loss of policy
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.buffer.logprobs)
        loss = - log_probs * returns
        loss = torch.sum(loss)

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()