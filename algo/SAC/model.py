import torch
import numpy as np
from algo.TD3.model import TD3
from torch.distributions import Normal


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

    def train(self):
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
