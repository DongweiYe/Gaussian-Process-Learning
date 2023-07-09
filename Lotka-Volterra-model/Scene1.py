import numpy as np
import GPy

import matplotlib.pyplot as plt


######################################
############## Scenario 1 ############
######################################

### Parameters
TrainRatio = 0.4
DataSparsity = 0.5
NoiseOuput = False
NumDyn = 2

### Load data
preydata = np.load('data/x1.npy')
preddata = np.load('data/x2.npy')
timedata = np.load('data/time.npy')

num_data = preddata.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)

print(samplelist.shape)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(preydata[samplelist],axis=1),np.expand_dims(preddata[samplelist],axis=1)))


### loop for the estimation of each dynamic equation
for i in range(0,NumDyn):
    


### Compute hyperparameters from a GP of x(t)
