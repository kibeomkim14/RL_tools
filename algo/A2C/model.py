import numpy as np
import torch
from algo.ActorCritic.model import AC


class A2C(AC):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size):
        """
        This class implements Advantage Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic. Upon training actor network, loss is simply the
        expected value of the product of log probability of an action given and the advantage.
        Advantage is calculated as the average of n-step return - Value function given the state.
        """
        super().__init__(env, Net, learning_rate, disc_rate, batch_size)

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
