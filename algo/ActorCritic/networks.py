import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, dim_out),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)