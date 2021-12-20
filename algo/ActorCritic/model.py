import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ExpReplay
from collections import namedtuple
from torch.distributions import Categorical
from utils.Algorithm import Algorithm


class AC(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size):
        """
        This class implements Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic.
        :param env: openai gym environment.
        :param Net: neural network class from pytorch module.
        :param learning_rate: learning rate of optimizer.
        :param disc_rate: discount rate used to calculate return G.
        :param batch_size: specified batch size for training.
        """
        self.dim_in = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.critic = Net(self.dim_in, 1)
        self.actor = Net(self.dim_in, self.dim_out)

        self.gamma = disc_rate
        self.batch_size = batch_size
        self.transition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'next_state', 'dones'))

        self.buffer = ExpReplay(10000, self.transition)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        # Here, we assumes the action space is discrete with a limited dimension.
        # given the dimension, network outputs probability of each action and the distribution
        # is formed. Action is then sampled from that distribution
        x = torch.from_numpy(state.astype(np.float32))  # change to tensor
        prob = self.actor.forward(x)
        dist = Categorical(prob)

        # action sampled from defined distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log_prob of pi(a|s)
        return action.item(), log_prob

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def train(self):
        # calculate return of all times in the episode
        if self.buffer.__len__() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        logprobs = torch.tensor(batch.logprobs, requires_grad=True).view(self.batch_size, 1)
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate.
        advantage = rewards + self.gamma * (1 - dones) * self.critic(next_states) - self.critic(states)
        critic_loss = advantage.pow(2).sum()
        actor_loss = - logprobs * advantage.detach()
        actor_loss = torch.sum(actor_loss)

        # do gradient ascent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
