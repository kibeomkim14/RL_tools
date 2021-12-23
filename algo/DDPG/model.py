import numpy as np
import torch
from copy import deepcopy
from utils.buffer import ExpReplay
from collections import namedtuple
from torch.distributions import Normal
from algo.ActorCritic.model import AC


class DDPG(AC):
    def __init__(self, env, Net, learning_rate, disc_rate, sigma, batch_size):
        super().__init__(env, Net, learning_rate, disc_rate)
        self.dim_out = 0
        self.batch_size = batch_size
        self.action_std = torch.tensor([sigma])
        self.tau = 0.05

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer = ExpReplay(10000, self.transition)
        self.exploration_dist = Normal(torch.tensor([0.0]), self.action_std)

        # Unlike AC setting, we view network as action-state value network.
        self.actor = Net(self.dim_in, 1)
        self.critic = Net(self.dim_in + 1, 1)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def act(self, state):
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        return torch.clip(action + self.exploration_dist.sample(), -2.0, 2.0).detach().numpy()

    def soft_update(self, target, source):
        for target_param, param in zip(list(target.parameters()), list(source.parameters())):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

    def train(self):
        # calculate return of all times in the episode
        if self.buffer.len() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        # extract variables from sampled batch.
        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)
        next_actions = self.actor_target(next_states)

        # compute target
        y = rewards + self.gamma * (1 - dones) * self.critic_target(torch.hstack((next_states, next_actions)))
        advantage = self.critic(torch.hstack([states, actions])) - y.detach()
        critic_loss = advantage.pow(2).mean()

        # Get actor loss
        actor_loss = -self.critic(torch.hstack([states, self.actor(states)])).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)
