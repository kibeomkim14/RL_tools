from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim

from algo.utils import base, buffer
from torch.distributions import Categorical, MultivariateNormal


class AC(base.Algorithm):
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
            self.dim_out = 1

        self.critic = Critic(self.dim_in)
        self.actor = Actor(self.dim_in, self.dim_out)

        self.gamma = disc_rate
        self.transition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'next_state', 'dones'))

        self.buffer = buffer.ExpReplay(10000, self.transition)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        """
        Here, we assume the action space is discrete with a limited dimension.
        given the dimension, network outputs probability of each action and the distribution
        is formed. Action is then sampled from that distribution
        :param state:
        :return:
        """
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

    def update(self):
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




class A2C(AC):
    def __init__(self, env, Net, learning_rate, disc_rate):
        """
        This class implements Advantage Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic. Upon training actor network, loss is simply the
        expected value of the product of log probability of an action given and the advantage.
        Advantage is calculated as the average of n-step return - Value function given the state.
        """
        super().__init__(env, Net, learning_rate, disc_rate)

    def update(self):
        # calculate return of all times in the episode
        transitions = self.buffer.all()
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.buffer.len(), self.dim_in)
        rewards = torch.tensor(batch.reward).view(self.buffer.len(), 1)
        dones = torch.tensor(batch.dones).view(self.buffer.len(), 1).long()
        logprobs = torch.tensor(batch.logprobs, requires_grad=True).view(self.buffer.len(), 1)
        next_states = torch.tensor(batch.next_state).view(self.buffer.len(), self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate.
        advantage = rewards + self.gamma * (1 - dones) * self.critic(next_states) - self.critic(states)
        critic_loss = advantage.pow(2).sum()

        # Unlike Actor-Critic, we instead use advantage Q(s,a) - V(s) for
        # calculating actor loss. However, we used TD estimate of Q(s,a)
        # which uses only value function estimate.
        actor_loss = - logprobs * advantage.detach()
        actor_loss = torch.sum(actor_loss)

        # do gradient ascent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # After training, empty the memory
        self.buffer.all()


class A2C_GAE(A2C):
    def __init__(self, env, Net, learning_rate, disc_rate, decay_rate):
        """
        This class implements Advantage Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic. Upon training actor network, loss is simply the
        expected value of the product of log probability of an action given and the advantage.
        Advantage is calculated as the average of n-step return - Value function given the state.
        """
        super().__init__(env, Net, learning_rate, disc_rate)
        self.lambda_ = decay_rate

    def update(self):
        # calculate return of all times in the episode
        transitions = self.buffer.all()
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.buffer.len(), self.dim_in)
        rewards = torch.tensor(batch.reward).view(self.buffer.len(), 1)
        dones = torch.tensor(batch.dones).view(self.buffer.len(), 1).long()
        logprobs = torch.tensor(batch.logprobs, requires_grad=True).view(self.buffer.len(), 1)
        next_states = torch.tensor(batch.next_state).view(self.buffer.len(), self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate.
        advantage   = rewards + self.gamma * (1 - dones) * self.critic(next_states) - self.critic(states)
        critic_loss = advantage.pow(2).sum()

        # generate a tensor of geometric series of weight coefficient
        # which is a product of discount rate and decay rate
        weight_coeff  = self.gamma * self.lambda_
        weight_tensor = torch.tensor([[weight_coeff ** i] for i in range(self.buffer.len())])
        inv_wgt_tensor = 1/weight_tensor
        weighted_adv  = weight_tensor * advantage
        weighted_adv = torch.flip(weighted_adv, dims=[1,0])
        weighted_adv = torch.cumsum(weighted_adv, dim=0)
        weighted_adv = torch.flip(weighted_adv, dims=[1,0])

        # in this algorithm, we use generalized advantage estimated proposed by
        # Schulman. we introduced the decay rate lambda to control the estimation of
        # advantage function for every state s_t.
        gae = weighted_adv * inv_wgt_tensor
        actor_loss = -logprobs * gae.detach()
        actor_loss = torch.sum(actor_loss)

        # do gradient ascent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class PPO(A2C_GAE):
    def __init__(self, env, Net, learning_rate, disc_rate, decay_rate, epsilon):
        """
        This class implements Advantage Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic. Upon training actor network, loss is simply the
        expected value of the product of log probability of an action given and the advantage.
        Advantage is calculated as the average of n-step return - Value function given the state.
        """
        super().__init__(env, Net, learning_rate, disc_rate, decay_rate)
        self.clip_ratio  = epsilon
        self.actor_old = Net(self.dim_in, self.dim_out)

    def act(self, state):
        # Overall process is similar to other Actor-Critic based algorithms, but
        # PPO outputs an action using the old policy.
        x = torch.tensor(state.astype(np.float32))  # change to tensor
        prob = self.actor_old.forward(x) # action from old policy.
        dist = Categorical(prob)

        # action sampled from defined distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log(pi(a|s))
        return action.item(), log_prob

    def update(self):
        transitions = self.buffer.all()
        batch = self.transition(*zip(*transitions))

        states   = torch.tensor(batch.state).view(self.buffer.len(), self.dim_in)
        rewards  = torch.tensor(batch.reward).view(self.buffer.len(), 1)
        dones    = torch.tensor(batch.dones).view(self.buffer.len(), 1).long()
        logp_old = torch.tensor(batch.logprobs).view(self.buffer.len(), 1)
        next_states = torch.tensor(batch.next_state).view(self.buffer.len(), self.dim_in)

        def eval_(state):
            prob = self.actor.forward(state)  # action from old policy.
            dist = Categorical(prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return torch.unsqueeze(log_prob, 1)

        # calculate loss of policy
        # Below advantage function is the TD estimate.
        advantage   = rewards + self.gamma * (1 - dones) * self.critic(next_states) - self.critic(states)
        critic_loss = advantage.pow(2).sum()

        # GENERALIZED ADVANTAGE ESTIMATION
        # generate a tensor of geometric series of weight coefficient
        # which is a product of discount rate and decay rate
        weight_coeff  = self.gamma * self.lambda_
        weight_tensor = torch.tensor([[weight_coeff ** i] for i in range(self.buffer.len())])
        inv_wgt_tensor = 1/weight_tensor
        weighted_adv  = weight_tensor * advantage
        weighted_adv = torch.flip(weighted_adv, dims=[1, 0])
        weighted_adv = torch.cumsum(weighted_adv, dim=0)
        weighted_adv = torch.flip(weighted_adv, dims=[1, 0])
        gae = weighted_adv * inv_wgt_tensor

        # calculate pi'(a|s)/pi(a|s)
        logp = eval_(states)
        ratio = torch.exp(logp - logp_old.detach())

        # CLIPPED SURROGATE OBJECTIVE
        # Now calculate surrogate objective introduced in Schulman,2017.
        surr_obj1  = ratio * gae.detach()
        surr_obj2  = torch.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * gae.detach()
        actor_loss = -torch.min(surr_obj1, surr_obj2)
        actor_loss = torch.sum(actor_loss)

        # GRADIENT ASCENT AND DESCENT
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # transfer parameters to old policy.
        self.actor_old.load_state_dict(self.actor.state_dict())
