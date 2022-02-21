import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from algo.utils import base, buffer
from collections  import namedtuple


class DQN(base.Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size, **kwargs):
        self.env  = env
        self.QNet = Net(**kwargs)

        self.dim_in  = self.env.observation_space.shape[0]
        self.dim_out = self.env.action_space.n


        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer     = buffer.ExpReplay(40000, self.transition)
        self.optimizer  = optim.Adam(self.QNet.parameters(), lr=learning_rate)

        self.gamma      = disc_rate
        self.batch_size = batch_size
        self.loss_function = nn.HuberLoss()

    def act(self, state, epsilon):
        # select action based on epsilon-greedy policy.
        if random.random() > epsilon:
            x = torch.tensor(state.astype(np.float32))  # change to tensor
            Q_values = self.QNet(x)
            action = torch.argmax(Q_values).item() # greedy action
        else:
            action = self.env.action_space.sample() # exploration
        assert action.size(0) == self.dim_out, 'dimension of action is not equal to action space requirements.' 
        return action

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def update(self):
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
        y = rewards + self.gamma * (1 - dones) * torch.max(self.QNet(next_states), 1)[0]
        Q = self.QNet(states).gather(1, actions)
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



class DoubleDQN(DQN):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size, tau):
        super().__init__(env, Net, learning_rate, disc_rate, batch_size)
        self.QNet_target = deepcopy(self.QNet)
        self.tau = tau

    def soft_update(self, target, source):
        for target_param, param in zip(list(target.parameters()), list(source.parameters())):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

    def update(self):
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
        next_actions = torch.max(self.QNet(next_states), 1)[1].view(self.batch_size, 1)

        # r + gamma Q(s_t+1, argmax_(a_t+1)(Q(s_t+1, a_t+1))
        y = rewards + self.gamma * (1 - dones) * self.QNet_target(states).gather(1, next_actions)
        Q = self.QNet(states).gather(1, actions) # Q(s_t, a_t)
        loss = (y - Q).pow(2).sum()

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        self.soft_update(self.QNet_target, self.QNet)

    def save(self, filename):
        torch.save(self.QNet.state_dict(), filename + "_DoubleDQN_Q")
        torch.save(self.QNet_target.state_dict(), filename + "_DoubleDQN_Q_target")
        torch.save(self.optimizer.state_dict(), filename + "_DQN_optimizer")

    def load(self, filename):
        self.QNet.load_state_dict(torch.load(filename + "_DoubleDQN_Q"))
        self.QNet_target.load_state_dict(torch.load(filename + "_DoubleDQN_Q_target"))
        self.optimizer.load_state_dict(torch.load(filename + "_DQN_optimizer"))


class DoubleDQN_PER(DoubleDQN):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size, tau, alpha):
        super().__init__(env, Net, learning_rate, disc_rate, batch_size, tau)
        self.alpha  = alpha
        self.buffer = buffer.PrioritizedExpReplay(40000, self.transition)
        self.delta  = 0

    def update(self):
        # if the number of sample collected is lower than the batch size,
        # we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, we begin training.
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))
        weights = self.buffer.calculate_weight()

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        next_actions = torch.max(self.QNet(next_states), 1)[1].view(self.batch_size, 1)

        # r + gamma Q(s_t+1, argmax_(a_t+1)(Q(s_t+1, a_t+1))
        TD_error = rewards + self.gamma * (1 - dones) * self.QNet_target(states).gather(1, next_actions) - self.QNet(states).gather(1, actions) # Q(s_t, a_t)
        self.buffer.update_priority(torch.abs(TD_error.detach()))
        loss = weights * TD_error.detach() * self.QNet(states).gather(1, actions)

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        self.soft_update(self.QNet_target, self.QNet)




