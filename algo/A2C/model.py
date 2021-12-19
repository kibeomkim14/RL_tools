import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ExpReplay

from torch.distributions import Categorical

gamma = 0.99  # discount rate


class A2C(nn.Module):
    def __init__(self, env, Net, learning_rate, disc_rate):
        super(A2C, self).__init__()

        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.critic  = Net(self.dim_in, 1)
        self.actor   = Net(self.dim_in, self.dim_out)

        self.gamma  = disc_rate
        self.buffer = ExpReplay(10000)
        self.actor_optimizer  = optim.Adam(self.actor.parameters() , lr=learning_rate)
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
        rewards     = torch.stack(self.buffer.rewards)
        next_states = torch.stack(self.buffer.next_states)
        states      = torch.stack(self.buffer.states)
        log_probs   = torch.stack(self.buffer.logprobs)

        # calculate loss of policy
        advantage = rewards + self.gamma * (1 - self.buffer.dones) * self.QNet(next_states) - self.QNet(states)
        actor_loss  = - log_probs * advantage.detach()
        actor_loss  = torch.sum(actor_loss)
        critic_loss = advantage.pow(2)

        # do gradient ascent
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
