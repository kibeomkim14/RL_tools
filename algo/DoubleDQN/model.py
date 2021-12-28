import torch
from copy import deepcopy
from algo.DQN.model import DQN


class DoubleDQN(DQN):
    def __init__(self, env, Net, learning_rate, disc_rate, epsilon, batch_size, tau):
        super().__init__(env, Net, learning_rate, disc_rate, epsilon, batch_size)
        self.QNet_target = deepcopy(self.QNet)
        self.target_tau = tau

    def soft_update(self, target, source):
        for target_param, param in zip(list(target.parameters()), list(source.parameters())):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

    def train(self):
        # if the number of sample collected is lower than the batch size,
        # we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, we begin training.
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        next_actions = torch.max(self.QNet_target(next_states), 1)[0]

        # r + gamma Q(s_t+1, argmax_(a_t+1)(Q(s_t+1, a_t+1))
        y = rewards + self.gamma * (1 - dones) * torch.index_select(self.QNet(states), 1, next_actions)
        Q = torch.index_select(self.QNet(states), 1, actions) # Q(s_t, a_t)
        loss = (y - Q).pow(2).sum()

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        self.soft_update(self.critic_target, self.critic)

    def save(self, filename):
        torch.save(self.QNet.state_dict(), filename + "_DoubleDQN_Q")
        torch.save(self.QNet_target.state_dict(), filename + "_DoubleDQN_Q_target")
        torch.save(self.optimizer.state_dict(), filename + "_DQN_optimizer")

    def load(self, filename):
        self.QNet.load_state_dict(torch.load(filename + "_DoubleDQN_Q"))
        self.QNet_target.load_state_dict(torch.load(filename + "_DoubleDQN_Q_target"))
        self.optimizer.load_state_dict(torch.load(filename + "_DQN_optimizer"))

