import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils.buffer import DQNMemory


gamma = 0.99  # discount rate


class SARSA(nn.Module):
    def __init__(self, Env, Net, learning_rate, disc_rate, epsilon):
        super(SARSA, self).__init__()

        self.env = Env
        self.dim_in  = Env.observation_space.shape[0]
        self.dim_out = Env.action_space.n
        self.valueNet  = Net(self.dim_in, self.dim_out)

        self.gamma  = disc_rate
        self.buffer = DQNMemory()
        self.optimizer = optim.Adam(self.valueNet.parameters(), lr=learning_rate)
        self.epsilon = epsilon

    def reset(self):
        self.buffer.clear()

    def act(self, state):
        # implements epsilon-greedy policy
        if random.random() < self.epsilon:
            return self.env.action_space.sample() # random action between 0 and 1. depends on the environment
        else:
            x = torch.from_numpy(state.astype(np.float32))  # change to tensor
            prob = self.valueNet.forward(x)

        return

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