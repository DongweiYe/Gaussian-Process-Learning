import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()
        # 1d input -> 1 hidden layer with one node -> 1d output 

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1, 1).double()  #hidden layer
        self.fc2 = nn.Linear(1, 1, bias=False).double()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


class MSERddloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fi, di, Rdd):
        loss = torch.mm(torch.mm(torch.transpose(fi-di, 0, 1),Rdd),fi-di)
        return loss