import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()
        # 1d input -> 1 hidden layer with one node -> 1d output 

        ### Example 1
        # self.fc1 = nn.Linear(1, 1).double()  #hidden layer
        # self.fc2 = nn.Linear(1, 1, bias=False).double()

        ### Example 2 & 3
        self.fc1 = nn.Linear(1, 8).double()  #hidden layer
        self.fc2 = nn.Linear(8, 8).double()  #hidden layer
        self.fc3 = nn.Linear(8, 8).double()  #hidden layer
        self.fc4 = nn.Linear(8, 1).double()  #hidden layer

        

    def forward(self, x):
        ### Example 1
        # x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        
        # ### Example 2 & 3
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)

        return x


class MSERddloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fi, di, Rdd):
        loss = torch.mm(torch.mm(torch.transpose(fi-di, 0, 1),Rdd),fi-di)
        return loss