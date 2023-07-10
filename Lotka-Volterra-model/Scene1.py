import numpy as np
import GPy

import matplotlib.pyplot as plt
np.random.seed(5)


def computeKdd(xa, xb, paramemeters):

    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def computeKdu(xa, xb, paramemeters):

    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)




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
    
    print('Estimate parameters in equation: ',i)
    ### Build a GP to infer the hyperparameters for each dynamic equation
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)
    xtGP.Gaussian_noise.fix(1e-3)
    xtGP.optimize(messages=True, max_f_eval=1, max_iters=1e7)
    # xtGP.optimize_restarts(num_restarts=2)

    ### Compute hyperparameters from a GP of x(t)
    # GPvariance = xtkernel[0]
    # GPlengthscale = xtkernel[1]

    ### Construct the covariance matrix of equation (5)
    Kuu = xtkernel.K(Xtrain)
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0)
    Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)
    Kud = np.transpose(Kdu)
    invKuu = np.linalg.inv(Kuu)

    Rdd = np.linalg.inv(Kdd-Kdu@invKuu@Kud)
    Rud = -Rdd@Kdu@invKuu
    Rdu = Rud.T

    print(ytrain[:,0:1].shape)
    print(np.multiply(ytrain[:,0:1],ytrain[:,1:2]).shape)

    if i == 0:
        G = np.hstack((ytrain[:,0:1],np.multiply(ytrain[:,0:1],ytrain[:,1:2])))
    else:
        G = np.hstack((np.multiply(ytrain[:,0:1],ytrain[:,1:2]),ytrain[:,0:1]))

    mu = -np.linalg.inv(G.T@Rdd@G)@G.T@Rdu@ytrain[:,i:(i+1)]
    print(mu)


