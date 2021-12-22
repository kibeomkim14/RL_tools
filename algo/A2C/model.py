import torch
from algo.ActorCritic.model import AC


class A2C(AC):
    def __init__(self, env, Net, learning_rate, disc_rate):
        """
        This class implements Advantage Actor-Critic algorithm where one uses two
        deep neural networks, Actor & Critic. Upon training actor network, loss is simply the
        expected value of the product of log probability of an action given and the advantage.
        Advantage is calculated as the average of n-step return - Value function given the state.
        """
        super().__init__(env, Net, learning_rate, disc_rate)

    def train(self):
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

    def train(self):
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