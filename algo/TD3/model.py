import torch
import numpy as np
import torch.optim as optim
from algo.DDPG.model import DDPG
from copy import deepcopy


class TD3(DDPG):
    """
    With Value Net and Policy Net defined, we integrate two network to learn value function and
    policy given states respectively.
    """
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size, update_freq, update_num):
        super().__init__(env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size)

        # introduce another Q-network with its target network and its optimizer.
        # first Q-net is already introduced in AC algorithm and this feature was inherited.
        self.critic_2 = Critic(self.dim_in)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.update_num  = update_num
        self.policy_freq = update_freq
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
        return torch.clip(action + torch.clip(self.noise_dist.sample(), -0.5, 0.5), -2.0, 2.0).detach().numpy()

    def train(self):
        # if number of collected sample is not enough, we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, do 'update_num' of updates.
        for _ in range(self.update_num):

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
            next_actions = torch.clip(next_actions + torch.clip(self.noise_dist.sample(), -0.5, 0.5), -2.0, 2.0).detach().numpy()

            # compute target
            Q_targ_1 = self.critic_target(torch.hstack((next_states, next_actions)))
            Q_targ_2 = self.critic_2_target(torch.hstack((next_states, next_actions)))
            Q_target = torch.min([Q_targ_1, Q_targ_2])

            y = rewards + self.gamma * (1 - dones) * Q_target
            advantage_1 = self.critic(torch.hstack([states, actions]))   - y.detach()
            advantage_2 = self.critic_2(torch.hstack([states, actions])) - y.detach()

            critic_loss_1 = advantage_1.pow(2).mean()
            critic_loss_2 = advantage_2.pow(2).mean()

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