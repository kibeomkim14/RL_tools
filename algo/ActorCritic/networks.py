import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 32),
            nn.ReLU(),
            nn.Linear(32, dim_out),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, dim_in):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
