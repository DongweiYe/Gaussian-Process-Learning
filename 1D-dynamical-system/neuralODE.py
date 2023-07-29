import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO

class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()
        # 1d input -> 1 hidden layer with one node -> 1d output 

        ### Example 1
        # self.fc1 = nn.Linear(1, 1).double()  #hidden layer
        # self.fc2 = nn.Linear(1, 1, bias=False).double()

        ### Example 2 
        self.fc1 = nn.Linear(1, 1).double()  #hidden layer
        self.fc2 = nn.Linear(1, 1, bias=False).double()
        # self.fc1 = nn.Linear(1, 8).double()  #hidden layer
        # self.fc2 = nn.Linear(8, 8).double()  #hidden layer
        # self.fc3 = nn.Linear(8, 8).double()  #hidden layer
        # self.fc4 = nn.Linear(8, 1).double()  #hidden layer

        

    def forward(self, x):
        ### Example 1
        # x = torch.tanh(self.fc1(x))
        # x = self.fc2(x)
        
        
        # ### Example 2 & 3
        x = torch.sin(self.fc1(x))
        x = self.fc2(x)

        return x


class MSERddloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fi, di, Rdd):
        loss = torch.mm(torch.mm(torch.transpose(fi-di, 0, 1),Rdd),fi-di)
        return loss
    

class BayesNeuralODE(PyroModule):

    def __init__(self,inverseRdd,prior_scale=10):
        super().__init__()
        # 1d input -> 1 hidden layer with one node -> 1d output 

        self.invRdd = inverseRdd
        ### Example 0
        self.linear1 = PyroModule[nn.Linear](1, 1)
        self.linear1.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., prior_scale).expand([1]).to_event(1))

        self.linear2 = PyroModule[nn.Linear](1, 1, bias=False)
        self.linear2.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))


        ### Example 1 (consistent)
        # self.linear1 = PyroModule[nn.Linear](1, 1)
        # self.linear1.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))
        # self.linear1.bias = PyroSample(dist.Normal(0., prior_scale).expand([1]).to_event(1))

        # self.linear2 = PyroModule[nn.Linear](1, 1)
        # self.linear2.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))
        # self.linear2.bias = PyroSample(dist.Normal(0., prior_scale).expand([1]).to_event(1))

        # self.linear3 = PyroModule[nn.Linear](2, 1, bias=False)
        # self.linear3.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 2]).to_event(2))
        ### Example 1 (general)
        

        

    def forward(self, x,y=None):
        ### Example 0
        x = torch.sin(self.linear1(x))
        x = self.linear2(x)

        ### Example 1 (consistent)
        # x1 = torch.tanh(self.linear1(x))
        # x2 = torch.sin(self.linear2(x))

        # x = self.linear3(torch.hstack((x1,x2)))


        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(torch.squeeze(x), self.invRdd.float()),obs=y)

        return torch.squeeze(x)