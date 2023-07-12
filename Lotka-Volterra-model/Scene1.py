import numpy as np
import GPy
import matplotlib.pyplot as plt
from LotkaVolterra_model import *
np.random.seed(0)


######################################
############## Scenario 1 ############
######################################
### The covariance matrix is highly possible to singular if no noise is added
### This applies to Kdd, Kuu, Kud and Kdu

### Parameters
TrainRatio = 0.4
DataSparsity = 0.05
NoiseMean = 0
NoiseSTD = 0.2 ### STD
NumDyn = 2
PosteriorSample = 200

### Load data and add noise
x1 = np.load('data/x1.npy')
x2 = np.load('data/x2.npy')
timedata = np.load('data/time.npy')

preydata = x1 + np.random.normal(NoiseMean,NoiseSTD,x1.shape[0])
preddata = x2 + np.random.normal(NoiseMean,NoiseSTD,x2.shape[0])

num_data = preddata.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)
# samplelist = np.arange(0,8000,100)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(preydata[samplelist],axis=1),np.expand_dims(preddata[samplelist],axis=1)))
ytrain_backup = np.hstack((np.expand_dims(x1[samplelist],axis=1),np.expand_dims(x2[samplelist],axis=1)))

plt.plot(Xtrain,ytrain[:,0],'*',label='x1 dynamics')
plt.plot(Xtrain,ytrain[:,1],'*',label='x2 dynamics')
plt.legend()
plt.show()

### An addition for loop just to get proper G_i data from regression
ytrain_hat = []
for i in range(0,NumDyn):
    xtkernel_add = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
    xtGP_add = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel_add)
    xtGP_add.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP_add.optimize_restarts(num_restarts=2,verbose=False)
    ypred,yvar = xtGP_add.predict(Xtrain)
    ytrain_hat.append(ypred)

ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T



### loop for the estimation of each dynamic equation
para_mean = []
para_cova = []
kernellist = []
GPlist = []

for i in range(0,NumDyn):
    
    print('Estimate parameters in equation: ',i)
    ### Build a GP to infer the hyperparameters for each dynamic equation
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)

    # xtkernel.lengthscale.constrain_bounded(0.3149, 0.315, warning=False)
    # xtkernel.variance.constrain_bounded(4.465, 4.475, warning=False)
    # xtGP.Gaussian_noise.fix(0.2)

    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)

    kernellist.append(xtkernel)
    GPlist.append(xtGP)
    # ypred,yvar = xtGP.predict(np.expand_dims(timedata,axis=1))
    # if i == 0:
    #     plt.plot(timedata,x1,'.',alpha=0.2)
    # else:
    #     plt.plot(timedata,x2,'.',alpha=0.2)

    # plt.plot(timedata,ypred,'-')
    # plt.show()

    ### Compute hyperparameters from a GP of x(t)
    GPvariance = xtkernel[0]
    GPlengthscale = xtkernel[1]
    GPnoise = xtGP['Gaussian_noise'][0][0]
    print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

    ### Construct the covariance matrix of equation (5)
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                 ### invertable
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise ### Additional noise to make sure invertable
    Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)                                            ### not invertable
    Kud = Kdu.T                                                                      ### not invertable
    invKuu = np.linalg.inv(Kuu)                                                  

    ### If we could only assume that Kuu is invertalbe, then
    Rdd = np.linalg.inv(Kdd-Kdu@invKuu@Kud)
    # Rdd_1 = np.linalg.inv(Kdd) + np.linalg.inv(Kdd)@Kdu@np.linalg.inv(Kuu-Kud@np.linalg.inv(Kdd)@Kdu)@Kud@np.linalg.inv(Kdd)
    Rdu = -Rdd@Kdu@invKuu
    Rud = Rdu.T

    # print(np.linalg.cond(Kuu))

    if i == 0:
        G = np.hstack((ytrain_hat[:,0:1],np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,1:2])))
    else:
        G = np.hstack((np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,1:2]),ytrain_hat[:,1:2]))

    mu_mean = -np.linalg.inv(G.T@Rdd@G)@G.T@Rdu@ytrain[:,i:(i+1)]
    mu_covariance = np.linalg.inv(G.T@Rdd@G)
    para_mean.append(mu_mean)
    para_cova.append(mu_covariance)

print('Parameter mean:', para_mean)
print('Parameter covariance: ',para_cova)

### Prediction with marginalization
preylist_array = []
predlist_array = []

# print(np.expand_dims(timedata[:int(num_data*TrainRatio+1)],axis=1))

# plt.plot(timedata[:int(num_data*TrainRatio)],np.squeeze(prey_train_phase_mean))
# plt.plot(timedata[:int(num_data*TrainRatio)],np.squeeze(pred_train_phase_mean))
# plt.show()

for i in range(PosteriorSample):

    mu1 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[0]),para_cova[0],1))
    mu2 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[1]),para_cova[1],1))
    # print(mu1,mu2)
    ### LV other parameters

    x1_t0 = 1
    x2_t0 = 1

    # x1_t0 = np.random.normal(prey_train_phase_mean[-1],prey_train_phase_var[-1],1) ### Prey initial
    # x2_t0 = np.random.normal(pred_train_phase_mean[-1],pred_train_phase_var[-1],1) ### Predator intial

    dt = 1e-3
    T = 20

    preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[mu1[0],-mu1[1],mu2[0],-mu2[1]])
    if np.max(preylist) > 200:
        pass
    else:
        preylist_array.append(preylist)
        predlist_array.append(predatorlist)


preymean = np.mean(np.asarray(preylist_array),axis=0)
predmean = np.mean(np.asarray(predlist_array),axis=0)
preystd = np.std(np.asarray(preylist_array),axis=0)
predstd = np.std(np.asarray(predlist_array),axis=0)

### append the mean and variance for GPR and Bayesian
# preymean = np.append(np.squeeze(prey_train_phase_mean),preymean)
# predmean = np.append(np.squeeze(pred_train_phase_mean),predmean)
# preystd = np.append(np.squeeze(prey_train_phase_var),preystd)
# predstd = np.append(np.squeeze(pred_train_phase_var),predstd)

plt.figure(figsize=(17, 2))
plt.plot(timedata,x1,'-k',linewidth=3,label='groundtruth')
plt.plot(timedata,x2,'-k',linewidth=3)

plt.plot(timedata,preymean,'--',color='royalblue',linewidth=3,label='prey prediction')
plt.plot(timedata,predmean,'--',color='tab:orange',linewidth=3,label='predator prediction')
plt.fill_between(timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5)
plt.fill_between(timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5)

plt.scatter(Xtrain,ytrain[:,0],marker='X',s=80,color='royalblue',edgecolors='k',label='training data (prey)',zorder=2)
plt.scatter(Xtrain,ytrain[:,1],marker='X',s=80,color='darkorange',edgecolors='k',label='training data (predator)',zorder=2)

plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')

# plt.legend()
# plt.show()
plt.savefig('N'+str(int(NoiseSTD*1000))+'D'+str(int(DataSparsity*1000))+'.png',bbox_inches='tight')