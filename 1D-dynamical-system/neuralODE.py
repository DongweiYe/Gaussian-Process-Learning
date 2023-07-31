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

        ### Example: sin function (consistent)
        self.fc1 = nn.Linear(1, 1).double()  #hidden layer
        self.fc2 = nn.Linear(1, 1, bias=False).double()

        ### Example: sin function (general)
        # self.fc1 = nn.Linear(1, 8).double()  #hidden layer
        # self.fc2 = nn.Linear(8, 8).double()  #hidden layer
        # self.fc3 = nn.Linear(8, 8).double()  #hidden layer
        # self.fc4 = nn.Linear(8, 1).double()  #hidden layer

        

    def forward(self, x):

        ### Example: sin function (consistent)
        x = torch.sin(self.fc1(x))
        x = self.fc2(x)

        ### Example: sin function (general)

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
        
        ### Example: sin function (consistent)
        # self.linear1 = PyroModule[nn.Linear](1, 1)
        # self.linear1.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))
        # self.linear1.bias = PyroSample(dist.Normal(0., prior_scale).expand([1]).to_event(1))

        # self.linear2 = PyroModule[nn.Linear](1, 1, bias=False)
        # self.linear2.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 1]).to_event(2))


        ### Example: sin function (general)
        self.linear1 = PyroModule[nn.Linear](1, 10)
        self.linear1.weight = PyroSample(dist.Normal(0., prior_scale).expand([10, 1]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., prior_scale).expand([10]).to_event(1))

        self.linear2 = PyroModule[nn.Linear](10, 1, bias=False)
        self.linear2.weight = PyroSample(dist.Normal(0., prior_scale).expand([1, 10]).to_event(2))
        
        

    def forward(self, x,y=None):
        
        ### Example: sin function (consistent)
        # x = torch.sin(self.linear1(x))
        # x = self.linear2(x)

        ### Example: sin function (general)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)


        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(torch.squeeze(x), self.invRdd.float()),obs=y)

        return torch.squeeze(x)