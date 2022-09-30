import torch 
import torch.nn as nn
import torch.nn.functional as F


class t2v(nn.Module):
    def __init__(self, inputsize, outpusize):
        super(t2v, self).__init__()
        self.fc = nn.Linear(inputsize, outpusize)

    def forward(self, x):
        x = self.fc(x)
        return torch.sin(x)
        