import torch
import random
import numpy as np
import torch.optim as optim
from collections import namedtuple
from utils.buffer import ExpReplay
from utils.Algorithm import Algorithm


class SARSA(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate, epsilon ,batch_size):
        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n

        self.env  = env
        self.QNet  = Net(self.dim_in, self.dim_out)

        self.epsilon = epsilon
        self.gamma   = disc_rate
        self.batch_size = batch_size
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer     = ExpReplay(10000, self.transition)
        self.optimizer  = optim.Adam(self.QNet.parameters(), lr=learning_rate)

    def act(self, state):
        # select action based on epsilon-greedy policy.
        if random.random() > self.epsilon:
            x = torch.tensor(state.astype(np.float32))  # change to tensor
            Q_values = self.QNet(x)
            action = torch.argmax(Q_values).item()  # greedy action
        else:
            action = self.env.action_space.sample()  # exploration
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

        states  = torch.tensor(np.array(batch.state)).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones   = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(np.array(batch.next_state)).view(self.batch_size, self.dim_in)

        # We obtain next_actions by inputting next_states in our Q network.
        # But with no_grad enabled since we only need auto grad on for Q function
        with torch.no_grad():
            if random.random() > self.epsilon:  # epsilon greedy
                Q_values = self.QNet(next_states)
                next_actions = torch.argmax(Q_values, dim=1)
            else:
                next_actions = torch.tensor([self.env.action_space.sample() for _ in range(self.batch_size)])
            next_actions = next_actions.view(self.batch_size, 1)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        y = rewards + self.gamma * (1 - dones) * self.QNet(next_states).gather(1, next_actions)
        Q = self.QNet(states).gather(1, actions)
        loss = (y - Q).pow(2).mean()

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.QNet.state_dict(), filename + "_SARSA_Q")
        torch.save(self.optimizer.state_dict(), filename + "_SARSA_optimizer")

    def load(self, filename):
        self.QNet.load_state_dict(torch.load(filename + "_SARSA_Q"))
        self.optimizer.load_state_dict(torch.load(filename + "_SARSA_optimizer"))