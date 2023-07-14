import numpy as np
import GPy
import matplotlib.pyplot as plt
from dynamical_system import *
np.random.seed(0)


######################################
############## Scenario 1 ############
######################################
### The covariance matrix is highly possible to singular if no noise is added
### This applies to Kdd, Kuu, Kud and Kdu

### Parameters
TrainRatio = 1/3         ### Train/Test data split ratio
DataSparsity = 1         ### Note different from scenarios 1 and 2, here we take 100% of as total data we have
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0           ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
PosteriorSample = 200    ### posterior sampling numbers

### Load data and add noise
x = np.load('data/tanh_x_ic1.npy')
timedata = np.load('data/time.npy')

NoiseSTD = NoisePer*np.mean(x)
xdata = x + np.random.normal(NoiseMean,NoiseSTD,x.shape[0])

num_data = x.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)


### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.expand_dims(xdata[samplelist],axis=1)

plt.plot(Xtrain,ytrain,'*',label='x dynamics')
plt.legend()
plt.show()

# ### Build a GP to infer the hyperparameters for each dynamic equation 
# ### and proper G_i data from regression
# ytrain_hat = []
# kernellist = []
# GPlist = []

# for i in range(0,NumDyn):
#     xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
#     xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)

#     xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
#     xtGP.optimize_restarts(num_restarts=2,verbose=False)
    
#     ypred,yvar = xtGP.predict(Xtrain)
#     ytrain_hat.append(ypred)
    
#     kernellist.append(xtkernel)
#     GPlist.append(xtGP)
    
# ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T

# ### loop for the estimation of each dynamic equation
# para_mean = []
# para_cova = []


# for i in range(0,NumDyn):
    
#     print('Estimate parameters in equation: ',i)

#     ### Compute hyperparameters from a GP of x(t)
#     GPvariance = kernellist[i][0]
#     GPlengthscale = kernellist[i][1]
#     GPnoise = GPlist[i]['Gaussian_noise'][0][0]
#     print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

#     ### Construct the covariance matrix of equation (5)
#     if NoisePer == 0:
#         Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*1e-4
#         Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-4
#     else:
#         Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    ### invertable
#         Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise ### Additional noise to make sure invertable

#     Kdu = kernellist[i].dK_dX(Xtrain,Xtrain,0)                                                  ### not invertable
#     Kud = Kdu.T                                                                                 ### not invertable
#     invKuu = np.linalg.inv(Kuu)                                                  

#     ### If we could only assume that Kuu is invertalbe, then
#     Rdd = np.linalg.inv(Kdd-Kdu@invKuu@Kud)
#     Rdu = -Rdd@Kdu@invKuu
#     Rud = Rdu.T

#     # print(np.linalg.cond(Kuu))

#     if i == 0:
#         G = np.hstack((ytrain_hat[:,0:1],np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,1:2])))
#     else:
#         G = np.hstack((np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,1:2]),ytrain_hat[:,1:2]))

#     mu_mean = -np.linalg.inv(G.T@Rdd@G)@G.T@Rdu@ytrain[:,i:(i+1)]
#     mu_covariance = np.linalg.inv(G.T@Rdd@G)
#     para_mean.append(mu_mean)
#     para_cova.append(mu_covariance)

# print('Parameter mean:', para_mean)
# print('Parameter covariance: ',para_cova)

# np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
# np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

# ### Prediction with marginalization
# preylist_array = []
# predlist_array = []

# for i in range(PosteriorSample):

#     mu1 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[0]),para_cova[0],1))
#     mu2 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[1]),para_cova[1],1))

#     ### LV other parameters
#     x1_t0 = 1
#     x2_t0 = 1

#     dt = 1e-3
#     T = 20

#     preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[mu1[0],-mu1[1],mu2[0],-mu2[1]])
#     if np.max(preylist) > 20 or np.max(predatorlist) > 20:
#         pass
#     else:
#         preylist_array.append(preylist)
#         predlist_array.append(predatorlist)


# preymean = np.mean(np.asarray(preylist_array),axis=0)
# predmean = np.mean(np.asarray(predlist_array),axis=0)
# preystd = np.std(np.asarray(preylist_array),axis=0)
# predstd = np.std(np.asarray(predlist_array),axis=0)


# plt.figure(figsize=(17, 2))
# params = {
#             'axes.labelsize': 21,
#             'font.size': 21,
#             'legend.fontsize': 23,
#             'xtick.labelsize': 21,
#             'ytick.labelsize': 21,
#             'text.usetex': False,
#             'axes.linewidth': 2,
#             'xtick.major.width': 2,
#             'ytick.major.width': 2,
#             'xtick.major.size': 2,
#             'ytick.major.size': 2,
#         }
# plt.rcParams.update(params)


# plt.plot(timedata,x2,'-k',linewidth=3)

# plt.plot(timedata,preymean,'--',color='royalblue',linewidth=3,label=r'$x_1$ prediction')
# plt.plot(timedata,predmean,'--',color='tab:orange',linewidth=3,label=r'$x_2$ prediction')
# plt.fill_between(timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5)
# plt.fill_between(timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5)

# plt.scatter(Xtrain,ytrain[:,0],marker='X',s=80,color='royalblue',edgecolors='k',label='training data '+r'($x_1$)',zorder=2)
# plt.scatter(Xtrain,ytrain[:,1],marker='X',s=80,color='darkorange',edgecolors='k',label='training data '+r'($x_2$)',zorder=2)

# plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
# plt.plot(timedata,x1,'-k',linewidth=3,label='ground truth')

# if NoisePer == 0:
#     plt.ylim([-0.8,8])
# # plt.xlim([-1,20])
# # plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncols=3,frameon=False)
# # plt.show()
# # plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')