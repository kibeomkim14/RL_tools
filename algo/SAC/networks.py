import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, dim_out),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, dim_in):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ELU(alpha=0.5)
        )

    def forward(self, x):
        return self.model(x)
