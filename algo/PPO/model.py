import torch
import numpy as np
from torch.distributions import Categorical, MultivariateNormal
from algo.A2C.model import *


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

    def train(self):
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