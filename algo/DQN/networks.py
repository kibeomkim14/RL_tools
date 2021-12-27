import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ValueNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 8),
            nn.Linear(8, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
