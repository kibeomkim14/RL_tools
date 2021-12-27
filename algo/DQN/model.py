import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

from utils.Algorithm import Algorithm
from collections import namedtuple
from utils.buffer import ExpReplay


class DQN(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate, epsilon, batch_size):
        self.env = env
        self.dim_in  = self.env.observation_space.shape[0]
        self.dim_out = self.env.action_space.n

        self.QNet = Net(self.dim_in, self.dim_out)

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer = ExpReplay(10000, self.transition)
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=learning_rate)

        self.gamma      = disc_rate
        self.epsilon    = epsilon
        self.batch_size = batch_size

    def act(self, state):
        # select action based on epsilon-greedy policy.
        if random.random() > self.epsilon:
            x = torch.tensor(state.astype(np.float32))  # change to tensor
            Q_values = self.QNet(x)
            action = torch.argmax(Q_values).item() # greedy action
        else:
            action = self.env.action_space.sample() # exploration
        return action

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def train(self):
        # if the number of sample collected is lower than the batch size,
        # we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, we begin training.
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        y = rewards + self.gamma * (1 - dones) * torch.max(self.critic(next_states), 1)[0]
        Q = torch.index_select(self.QNet(states), 1, actions)
        loss = (y - Q).pow(2).sum()

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.QNet.state_dict(), filename + "_DQN_Q")
        torch.save(self.optimizer.state_dict(), filename + "_DQN_optimizer")

    def load(self, filename):
        self.QNet.load_state_dict(torch.load(filename + "_DQN_Q"))
        self.optimizer.load_state_dict(torch.load(filename + "_DQN_optimizer"))
