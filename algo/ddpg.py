import numpy as np
import torch
from copy import deepcopy
from algo.utils import buffer, base
from collections import namedtuple
from torch.distributions import Normal
from algo.ac import AC
import torch.optim as optim

class DDPG(AC):
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size):
        super().__init__(env, Actor, Critic, learning_rate, disc_rate)
        self.dim_out = 0
        self.batch_size = batch_size
        self.tau = 0.05

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer = buffer.ExpReplay(10000, self.transition)
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([sigma]))

        self.actor_target  = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def act(self, state):
        """
        Unlike actor-critic methods, we use policy function to find
        the deterministic action instead of distributions which is parametrized by
        distribution parameters learned from the policy.

        Here, state input prompts policy network to output a single or multiple-dim
        actions.
        :param state:
        :return:
        """
        # we
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        return torch.clip(action + self.noise_dist.sample(), -2.0, 2.0).detach().numpy()

    def soft_update(self, target, source):
        for target_param, param in zip(list(target.parameters()), list(source.parameters())):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

    def update(self):
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


class TD3(DDPG):
    """
    With Value Net and Policy Net defined, we integrate two network to learn value function and
    policy given states respectively.
    """
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size, policy_update_freq, update_num):
        super().__init__(env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size)
        # introduce another Q-network with its target network and its optimizer.
        # first Q-net is already introduced in AC algorithm and this feature was inherited.
        self.critic = Critic(self.dim_in + 1)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_2 = Critic(self.dim_in + 1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.update_num  = update_num
        self.policy_freq = policy_update_freq
        self.update_time = 0

    def act(self, state):
        """
        on top of DDPG act method, we introduce extra clip on noise sampled from distribution. 
        Then the clipped noise is added to action from the policy network.
        :param state:
        :return:
        """
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        return torch.clip(action + torch.clip(self.noise_dist.sample(), -0.5, 0.5), -2.0, 2.0).detach()

    def update(self):
        # if number of collected sample is not enough, we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, do 'update_num' of updates.
        for j in range(self.update_num):

            transitions = self.buffer.sample(self.batch_size)
            batch = self.transition(*zip(*transitions))

            # extract variables from sampled batch.
            states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
            actions = torch.tensor(batch.action).view(self.batch_size, 1)
            rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
            dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
            next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

            # compute target actions with target network.
            next_actions = self.actor_target(next_states)
            next_actions = torch.clip(next_actions + torch.clip(self.noise_dist.sample(), -0.5, 0.5), -2.0, 2.0).detach()

            # compute target
            Q_targ_1 = self.critic_target(torch.hstack((next_states, next_actions)))
            Q_targ_2 = self.critic_2_target(torch.hstack((next_states, next_actions)))
            Q_target = torch.minimum(Q_targ_1, Q_targ_2)

            y = rewards + self.gamma * (1 - dones) * Q_target
            advantage_1 = self.critic(torch.hstack([states, actions]))   - y.detach()
            advantage_2 = self.critic_2(torch.hstack([states, actions])) - y.detach()

            critic_loss_1 = advantage_1.pow(2).mean()
            critic_loss_2 = advantage_2.pow(2).mean()

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_loss_2.backward()
            self.critic_2_optimizer.step()

            if j % self.policy_freq == 0:
                # Get actor loss
                actor_loss = -self.critic(torch.hstack([states, self.actor(states)])).mean()

                # Update the frozen target models
                self.soft_update(self.critic_target  , self.critic)
                self.soft_update(self.critic_2_target, self.critic_2)
                self.soft_update(self.actor_target, self.actor)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()



class SAC(TD3):
    """
    With Value Net and Policy Net defined, we integrate two network to learn value function and
    policy given states respectively.
    """
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size,
                 policy_update_freq, update_num, alpha):
        super().__init__(env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size, policy_update_freq, update_num)
        # introduce another Q-network with its target network and its optimizer.
        # first Q-net is already introduced in AC algorithm and this feature was inherited.
        self.alpha = alpha
        self.action_var = sigma

    def act(self, state):
        """
        on top of DDPG act method, we introduce extra clip on noise sampled from distribution. 
        Then the clipped noise is added to action from the policy network.
        :param state:
        :return:
        """
        x = torch.tensor(state.astype(np.float32))
        mu_ = self.actor.forward(x)
        dist = Normal(mu_, self.action_var)
        action = dist.sample()
        return action.detach()

    def update(self):
        # if number of collected sample is not enough, we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, do 'update_num' of updates.
        for j in range(self.update_num):

            transitions = self.buffer.sample(self.batch_size)
            batch = self.transition(*zip(*transitions))

            # extract variables from sampled batch.
            states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
            actions = torch.tensor(batch.action).view(self.batch_size, 1)
            rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
            dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
            next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

            with torch.no_grad():
                # calculate log probability, log_{pi_theta}(a'|s')
                mu_ = self.actor(next_states)
                dist = Normal(mu_, self.action_var * torch.ones(mu_.size()))
                next_actions = dist.sample()
                logprobs = dist.log_prob(next_actions)

                # compute target
                Q_targ_1 = self.critic_target(torch.hstack((next_states, next_actions)))
                Q_targ_2 = self.critic_2_target(torch.hstack((next_states, next_actions)))
                Q_target = torch.minimum(Q_targ_1, Q_targ_2)

            # calculate critic loss
            y = rewards + self.gamma * (1 - dones) * (Q_target - self.alpha * logprobs)
            advantage_1 = self.critic(torch.hstack([states, actions]))   - y.detach()
            advantage_2 = self.critic_2(torch.hstack([states, actions])) - y.detach()

            critic_loss_1 = advantage_1.pow(2).mean()
            critic_loss_2 = advantage_2.pow(2).mean()

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_loss_2.backward()
            self.critic_2_optimizer.step()


            # calculate log probability, log_{pi_theta}(a'|s')
            mu_ = self.actor(states)
            dist = Normal(mu_, self.action_var * torch.ones(mu_.size()))
            actions_theta = dist.rsample() # used rsample to indicate this is due to reparametrization trick
            logprobs = dist.log_prob(actions_theta)

            Q_1 = self.critic(torch.hstack((states, actions_theta)))
            Q_2 = self.critic_2(torch.hstack((states, actions_theta)))
            Q = torch.minimum(Q_1, Q_2)

            actor_loss = -(Q - self.alpha * logprobs).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if j % self.policy_freq == 0:
                # Update the frozen target models
                self.soft_update(self.critic_target  , self.critic)
                self.soft_update(self.critic_2_target, self.critic_2)
