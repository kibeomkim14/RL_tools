import torch
import numpy as np
from algo.DDPG.model import DDPG
from copy import deepcopy


class TD3(DDPG):
    '''
    With Value Net and Policy Net defined, we integrate two network to learn value function and
    policy given states respectively.
    '''
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size, update_freq):
        super().__init__(env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size)
        self.policy_freq = update_freq
        self.update_time = 0

    def act(self, state)
        """
        on top of DDPG act method, we introduce extra clip on
        noise sampled from distribution. Then the clipped noise is added to action from
        policy network.
        :param state:
        :return:
        """
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        return torch.clip(action + torch.clip(self.noise_dist.sample(), -0.5, 0.5), -2.0, 2.0).detach().numpy()

    def train(self):
        # take B samples from experience replay memory
        # NOTE: they are formed as torch.Tensor
        self.update_time += 1

        states, actions, rewards, next_states, dones = self.exp_memory.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.noise_std / 2).clamp(-self.noise_clip, self.noise_clip)
            next_actions = torch.FloatTensor(np.clip(self.actor_target(next_states) + noise.detach().numpy(), 0, 1))

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards.view(self.batch_size, 1) + dones.sub(1).mul(-1) * self.gamma * target_Q.view(
                self.batch_size, 1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1.view(self.batch_size, 1), target_Q) + F.mse_loss(
            current_Q2.view(self.batch_size, 1), target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.update_time % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.actor_target, self.actor)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = deepcopy(self.actor)