import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)



class CNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, kernel=2, stride=1, dropout:float=0.1):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,2,),
            nn.ReLU(),
            nn.Linear(64, dim_out),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)

