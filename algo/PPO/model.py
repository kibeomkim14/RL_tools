import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from utils.buffer import ExpReplay
from torch.distributions import Categorical, MultivariateNormal


class PPO(object):
    def __init__(self, env, Net, actor_hidden_dim, critic_hidden_dim, ppo_epoch,
                 disc_rate, c1_coef, c2_coef, actor_variance,  epsilon = 0.2, lr_critic = 1e-4,
                 lr_actor = 1e-4):

        # Set two separate neural networks and its target network
        self.env    = env
        self.critic = Net(self.env.obs_space.shape[0], critic_hidden_dim, 1)
        self.actor  = Net(self.env.obs_space.shape[0], actor_hidden_dim , env.action_space.n)

        self.actor_old = deepcopy(self.actor)
        self.c1_coef   = c1_coef
        self.c2_coef   = c2_coef
        self.VF_loss   = nn.MSELoss()
        self.gamma     = disc_rate

        self.clip_epsilon  = epsilon
        self.ppo_epoch     = ppo_epoch
        self.action_var    = torch.full((self.env.act_space.shape[0],), actor_variance)
        self.exp_memory    = ExpReplay()

        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr_critic)
        self.actor_optimizer  = optim.SGD(self.actor.parameters() , lr_actor)

    def act(self, state):

        state    = torch.FloatTensor(state)
        mean     = self.actor_old(state)
        cov_mat  = torch.diag(self.action_var).unsqueeze(dim=0)
        dist     = MultivariateNormal(mean, cov_mat)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        clipper  = nn.Softmax(dim=1)

        return clipper(action.detach()), log_prob.detach()

    def eval_(self, state, action):

        mean = self.actor(state)
        var  = self.action_var.expand_as(mean)
        cov_mat = torch.diag_embed(var)
        dist    = MultivariateNormal(mean, cov_mat)

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state)

        return logprob, state_values, entropy

    def store(self, action, state, logprob, reward, done):
        '''
        '''
        if type(state) != torch.Tensor:
            state = torch.Tensor(state)

        self.exp_memory.store(state, action, logprob, reward, done)

    def train(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.exp_memory.rewards), reversed(self.exp_memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states   = torch.squeeze(torch.stack(self.exp_memory.states  , dim=0)).detach()
        old_actions  = torch.squeeze(torch.stack(self.exp_memory.actions , dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.exp_memory.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.ppo_epoch):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.eval_(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            L_clip1    = ratios * advantages
            L_clip2    = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # final loss of clipped objective PPO
            actor_loss  = - torch.min(L_clip1, L_clip2) - self.c2_coef * dist_entropy
            critic_loss = self.c1_coef * self.VF_loss(state_values, rewards)

            # take gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()


        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        # clear buffer
        self.exp_memory.clear()

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
