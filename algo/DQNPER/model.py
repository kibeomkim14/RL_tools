import torch
from collections import namedtuple
from utils.buffer import PrioritizedExpReplay
from algo.DoubleDQN.model import DoubleDQN


class DoubleDQN_PER(DoubleDQN):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size, tau, alpha):
        super().__init__(env, Net, learning_rate, disc_rate, batch_size, tau)
        self.alpha  = alpha
        self.buffer = PrioritizedExpReplay(40000, self.transition)
        self.delta  = 0

    def train(self):
        # if the number of sample collected is lower than the batch size,
        # we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, we begin training.
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))
        weights = self.buffer.calculate_weight()

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        next_actions = torch.max(self.QNet(next_states), 1)[1].view(self.batch_size, 1)

        # r + gamma Q(s_t+1, argmax_(a_t+1)(Q(s_t+1, a_t+1))
        TD_error = rewards + self.gamma * (1 - dones) * self.QNet_target(states).gather(1, next_actions) - self.QNet(states).gather(1, actions) # Q(s_t, a_t)
        self.buffer.update_priority(torch.abs(TD_error.detach()))
        loss = weights * TD_error.detach() * self.QNet(states).gather(1, actions)

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        self.soft_update(self.QNet_target, self.QNet)

