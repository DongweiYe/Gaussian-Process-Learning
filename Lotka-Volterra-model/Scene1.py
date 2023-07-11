import numpy as np
import GPy
import matplotlib.pyplot as plt
np.random.seed(1)


######################################
############## Scenario 1 ############
######################################
### The covariance matrix is highly possible to singular if no noise is added
### This applies to Kdd, Kuu, Kud and Kdu

### Parameters
TrainRatio = 0.4
DataSparsity = 0.01
NoiseLevel = 1
NumDyn = 2


### Load data and add noise
x1 = np.load('data/x1.npy')
x2 = np.load('data/x2.npy')
timedata = np.load('data/time.npy')

preydata = x1 + np.random.rand(x1.shape[0])*NoiseLevel
preddata = x2 + np.random.rand(x2.shape[0])*NoiseLevel

num_data = preddata.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)
# samplelist = np.arange(0,8000,5)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(preydata[samplelist],axis=1),np.expand_dims(preddata[samplelist],axis=1)))

plt.plot(Xtrain,ytrain[:,0],'*')
plt.plot(Xtrain,ytrain[:,1],'*')
plt.show()

### loop for the estimation of each dynamic equation
for i in range(0,NumDyn):
    
    print('Estimate parameters in equation: ',i)
    ### Build a GP to infer the hyperparameters for each dynamic equation
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)
    # xtGP.Gaussian_noise.fix(2)
    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)

    ypred,yvar = xtGP.predict(np.expand_dims(timedata,axis=1))
    if i == 0:
        plt.plot(timedata,x1,'.',alpha=0.2)
    else:
        plt.plot(timedata,x2,'.',alpha=0.2)

    plt.plot(timedata,ypred,'-')
    plt.show()

    ### Compute hyperparameters from a GP of x(t)
    GPvariance = xtkernel[0]
    GPlengthscale = xtkernel[1]
    GPnoise = xtGP['Gaussian_noise'][0][0]
    print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

    ### Construct the covariance matrix of equation (5)
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*1e-4             ### invertable
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-3  ### Additional noise to make sure invertable
    Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)                                  ### not invertable
    Kud = Kdu.T                                                                  ### not invertable
    invKuu = np.linalg.inv(Kuu)                                                  

    ### If we could only assume that Kuu is invertalbe, then
    Rdd = np.linalg.inv(Kdd-Kdu@invKuu@Kud)
    # Rdd_1 = np.linalg.inv(Kdd) + np.linalg.inv(Kdd)@Kdu@np.linalg.inv(Kuu-Kud@np.linalg.inv(Kdd)@Kdu)@Kud@np.linalg.inv(Kdd)
    Rdu = -Rdd@Kdu@invKuu
    Rud = Rdu.T

    print(np.linalg.cond(Kuu))
    # print(Rdd-Rdd_1)

    if i == 0:
        G = np.hstack((ytrain[:,0:1],np.multiply(ytrain[:,0:1],ytrain[:,1:2])))
    else:
        G = np.hstack((np.multiply(ytrain[:,0:1],ytrain[:,1:2]),ytrain[:,1:2]))

    mu = -np.linalg.inv(G.T@Rdd@G)@G.T@Rdu@ytrain[:,i:(i+1)]
    print(mu)


