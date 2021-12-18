import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from utils.buffer import DQNMemory, Transition

BATCHSIZE = 100

class DQN(nn.Module):
    def __init__(self, env, Net, learning_rate, disc_rate, epsilon):
        super(DQN, self).__init__()

        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.QNet    = Net(self.dim_in, self.dim_out)
        self.QTarget = deepcopy(self.QNet)

        self.gamma   = disc_rate
        self.buffer  = DQNMemory(10000) # exp memory of size 10,000.
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.batch_size = BATCHSIZE

    def reset(self):
        self.buffer.clear()

    def act(self, state):
        # DQN takes epsilon-greedy policy, which is for given epsilon, the agent acts randomly
        # otherwise, the agent takes an action that maximizes Q-value given state input.
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.QNet(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.dim_out)]], dtype=torch.long)

    def store(self, state, action, reward, next_state, done):
        """
        stores the output of environment and agents into agents experience buffer.
        :return:
        """
        self.buffer.store(state, action, reward, next_state, done)

    def train(self):

        # check if memory has enough samples to train
        if self.buffer.__len__() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # calculate return of all times in the episode
        reward_list = self.buffer.rewards
        T = len(reward_list)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        # calculate returns recursively
        for t in reversed(range(T)):
            future_return = reward_list[t] + self.gamma * future_return
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