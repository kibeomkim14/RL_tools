import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ValueNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_out)
        )

    def forward(self, x):
        return self.model(x)
