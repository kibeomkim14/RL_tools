import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from utils.buffer import ExpReplay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_noise(action, noise_std):
    noise = np.random.normal(0 ,noise_std, len(action))
    action = action + noise

    if action.min() < 0:
        action = action + np.abs(action.min())
    action = action /np.sum(action)
    return action


class TD3(object):
    '''
    With Value Net and Policy Net defined, we integrate two network to learn value function and
    policy given states respectively.
    '''
    def __init__(self, env, Net, gamma, actor_hidden_dim, critic_hidden_dim,
                 seq_length, lr_critic = 1e-6, lr_actor = 1e-3, policy_freq = 2,
                 tau = 0.05, memory_size = 500, batch_size = 50, noise_std =0.2,
                 noise_clip = 0.5):

        self.env = env
        self.observ_dim  = env.observation_space.shape[0]
        self.action_dim  = env.action_space.shape[0]
        self.seq_len     = seq_length
        self.gamma       = gamma
        self.tau         = tau

        self.exp_memory  = ExpReplay(10000)
        self.memory_size = memory_size
        self.batch_size  = batch_size
        self.update_time = 0
        self.policy_freq = policy_freq
        self.noise_std   = noise_std
        self.noise_clip  = noise_clip

        # Set two separate neural networks and its target network
        self.critic = Net(self.observ_dim + self.action_dim, critic_hidden_dim)
        self.actor  = Net(self.observ_dim, actor_hidden_dim, self.action_dim)

        self.critic_target = deepcopy(self.critic)
        self.actor_target  = deepcopy(self.actor)

        # self.critic_h0, self.critic_c0  = self.critic.init_state('zero')
        # self.critic_target_h0, self.critic_target_c0  = self.critic_target.init_state('zero')

        self.critic_optimizer = optim.Adam(self.critic.parameters() , lr_critic)
        self.actor_optimizer  = optim.Adam(self.actor.parameters()  , lr_actor)

    def act(self, state):
        state  = torch.FloatTensor(state).to(device)
        action = self.actor(state).detach().numpy()
        return add_noise(action, self.noise_std)

    def eval_(self, state):
        state  = torch.FloatTensor(state).to(device)
        return self.actor(state).detach().numpy()

    def store(self, *args):
        self.exp_memory.store(*args)

    def reset(self):
        self.exp_memory.clear()

    def soft_update(self, target, source):
        for target_param, param in zip(list(target.parameters()), list(source.parameters())):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

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