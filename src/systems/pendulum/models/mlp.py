import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(2,128),
        nn.Tanh(),
        nn.Linear(128,128),
        nn.Tanh(),
        nn.Linear(128,2)
    )

  def forward(self,x):
    return self.net(x)
