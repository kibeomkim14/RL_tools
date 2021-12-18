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

        # sample transition from experience memory.
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # tensor-ize batch samples.
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.QNet(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.QTarget(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.QNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()