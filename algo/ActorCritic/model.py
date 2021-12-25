from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from utils.Algorithm import Algorithm
from utils.buffer import ExpReplay


class AC(Algorithm):
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate):

        """
        This class implements 1-step Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic.
        :param env: openai gym environment.
        :param Net: neural network class from pytorch module.
        :param learning_rate: learning rate of optimizer.
        :param disc_rate: discount rate used to calculate return G
        """
        self.dim_in = env.observation_space.shape[0]
        try:
            self.dim_out = env.action_space.n
        except:
            self.dim_out = 0

        self.critic = Critic(self.dim_in)
        self.actor = Actor(self.dim_in, self.dim_out)

        self.gamma = disc_rate
        self.transition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'next_state', 'dones'))

        self.buffer = ExpReplay(10000, self.transition)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        # Here, we assumes the action space is discrete with a limited dimension.
        # given the dimension, network outputs probability of each action and the distribution
        # is formed. Action is then sampled from that distribution
        x = torch.tensor(state.astype(np.float32))  # change to tensor
        prob = self.actor.forward(x)
        dist = Categorical(prob)

        # action sampled from defined distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log(pi(a|s))
        return action.item(), log_prob

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def train(self):
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.buffer.len())
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.buffer.len(), self.dim_in)
        rewards = torch.tensor(batch.reward).view(self.buffer.len(), 1)
        dones = torch.tensor(batch.dones).view(self.buffer.len(), 1).long()
        logprobs = torch.tensor(batch.logprobs, requires_grad=True).view(self.buffer.len(), 1)
        next_states = torch.tensor(batch.next_state).view(self.buffer.len(), self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        Q = rewards + self.gamma * (1 - dones) * self.critic(next_states)
        advantage = Q - self.critic(states)
        critic_loss = advantage.pow(2).sum()

        # Although training process is similar to REINFORCE algorithm
        # in Actor-Critic, return, G is substituted for Q(s,a)
        # In this code, we calculated TD estimate of Q(s,a)
        actor_loss = - logprobs * Q.detach()
        actor_loss = torch.sum(actor_loss)

        # do gradient ascent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
