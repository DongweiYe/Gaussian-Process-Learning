import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
from dynamical_system import *
# from neuralODE import *

import pymc as pm
import pytensor as pyte

np.random.seed(0)

########################################
############## Scenario 2 ##############
########################################

### Parameters
TrainRatio = 0.2         ### Train/Test data split ratio
NumDataTrain = 1000      ### Number of data for training
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0.1         ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
ShapeNN = [1,8,1]        ### nodes of each layer of a neural network, e.g. [1,8,8,1]
PosteriorSample = 1000    ### number of posterior samples for each core from MCMC
num_ic = 3

### Load data and add noise
x_list = []
for i in range(num_ic):
    x_list.append(np.load('data/ODEdata_'+str(i)+'.npy'))
timedata = np.load('data/time.npy')

num_data = timedata.shape[0] - 1  ### 0 -> K, using [0,K-1] for int
num_train = NumDataTrain 

samplelist_list = []
Xtrain_list = []
NoiseSTD_list = []
for i in range(num_ic):
    samplelist_list.append(np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False))
    Xtrain_list.append(np.expand_dims(timedata[samplelist_list[i]],axis=1))
    NoiseSTD_list.append(NoisePer*x_list[i])
    # NoiseSTD_list.append(NoisePer*np.mean(x_list[i]))

xdata_list = []
ytrain_list = []


for i in range(num_ic):
    xdata_list.append(x_list[i] + np.squeeze(np.random.normal(NoiseMean*np.ones(x_list[i].shape[0])
                                                                ,NoiseSTD_list[i])))
    # xdata_list.append(x_list[i] + np.random.normal(NoiseMean,NoiseSTD_list[i],x_list[i].shape[0]))
    ytrain_list.append(np.expand_dims(xdata_list[i][samplelist_list[i]],axis=1))

    plt.plot(Xtrain_list[i],ytrain_list[i],'*',label='x data')
    plt.plot(timedata,x_list[i],'-k',label='x')
plt.legend()
plt.savefig('data.png')

print('Data from training: ',num_train*num_ic)


### Build a GP to infer the hyperparameters for each dynamic equation 
### and compute d_i conditioning on u_i

ytrain_hat_list = []
kernel_list = []
GP_list = []
plt.clf()
for i in range(num_ic):
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
    xtGP = GPy.models.GPRegression(Xtrain_list[i],ytrain_list[i],xtkernel)

    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)

    ytrain_hat,ytrain_hat_var = xtGP.predict(Xtrain_list[i])
    ytrain_hat_list.append(ytrain_hat)
    kernel_list.append(xtkernel)
    GP_list.append(xtGP)
    ### Vsiualization
    plt.plot(Xtrain_list[i],ytrain_list[i],'b*',label='GT')
    plt.plot(Xtrain_list[i],ytrain_hat,'o',color='tab:red',label='GP prediction',alpha=0.5)

plt.legend()
plt.savefig('GPprediction.png')

### Print out hyperparameters from a GP of x(t)
Kuu_list = []
Kdd_list = []
Kdu_list = []
Kud_list = []
Rdd_list = []
Rdu_list = []
invKuu_list = []
invRdd_list = []
d_hat_list = []

for i in range(num_ic):
    print('     Prepare IC data: ',i)

    GPvariance = kernel_list[i][0]
    GPlengthscale = kernel_list[i][1]
    GPnoise = GP_list[i]['Gaussian_noise'][0][0]
    print('     GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)  

    ### Construct the covariance matrix 
    if NoisePer == 0:
        Kuu_list.append(kernel_list[i].K(Xtrain_list[i]) + np.identity(Xtrain_list[i].shape[0])*1e-8)
        ### Note that the artifical noise here sometime may need to be scale up, e.g. 1e-5,1e-4, to make sure inRdd is invertable in pyMC with large dataset
        Kdd_list.append(kernel_list[i].dK2_dXdX2(Xtrain_list[i],Xtrain_list[i],0,0) + np.identity(Xtrain_list[i].shape[0])*1e-8)
    else:
        Kuu_list.append(kernel_list[i].K(Xtrain_list[i]) + np.identity(Xtrain_list[i].shape[0])*GPnoise)                    
        Kdd_list.append(kernel_list[i].dK2_dXdX2(Xtrain_list[i],Xtrain_list[i],0,0) + np.identity(Xtrain_list[i].shape[0])*GPnoise)

    Kdu_list.append(kernel_list[i].dK_dX(Xtrain_list[i],Xtrain_list[i],0))
    Kud_list.append(Kdu_list[i].T)
    invKuu_list.append(np.linalg.inv(Kuu_list[i]))

    invRdd_list.append(Kdd_list[i]-Kdu_list[i]@invKuu_list[i]@Kud_list[i])
    Rdd_list.append(np.linalg.inv(invRdd_list[i]))

    ### Compute the true value of d_i using GP
    d_hat_list.append(Kdu_list[i]@invKuu_list[i]@ytrain_list[i])

# print(d_hat_list)
# print(ytrain_hat_list)
##############################################
############# Build neural network ###########
##############################################

Y_obs_list = []
layer1_list = []
layer2_list = []
out_list = []
### Construct structure of NN to the dynamics systems
with pm.Model() as model:

    init_1 = np.random.normal(-1,0.1,size=(ShapeNN[0],ShapeNN[1]))
    init_2 = np.random.normal(3,0.1,size=(ShapeNN[1],ShapeNN[2]))
    ### Weights and bias from input to hidden layer
    layer1_weight = pm.Normal('l1_w', 0, sigma=10,initval=init_1, shape=(ShapeNN[0],ShapeNN[1]))
    layer1_bias = pm.Normal('l1_b', 0, sigma=10, shape=ShapeNN[1])
    layer2_weight = pm.Normal('l2_w', 0, sigma=10,initval=init_2, shape=(ShapeNN[1],ShapeNN[2]))

    ### Feedforward
    for i in range(num_ic):
        layer1_list.append(pm.math.tanh(pm.math.dot(ytrain_hat_list[i],layer1_weight) + layer1_bias))
        out_list.append(pm.math.dot(layer1_list[i],layer2_weight))
        Y_obs_list.append(pm.MvNormal('Y_obs_'+str(i), mu=out_list[i], cov=invRdd_list[i], observed=d_hat_list[i]))
    # layer1 = pm.Deterministic('a1',pm.math.tanh(layer1_weight*ytrain_hat+layer1_bias))
    # layer2 = pm.Deterministic('a2',layer2_weight*layer1)
    
    step = pm.Metropolis()
    trace = pm.sample(PosteriorSample,step=step, return_inferencedata=False,cores=4,tune=2000,random_seed=0)
    # approx = pm.fit(100000,method='fullrank_advi',random_seed=0) ### VI


posterior_samples_l1w = np.squeeze(trace.get_values("l1_w", combine=True))
posterior_samples_l1b = np.squeeze(trace.get_values("l1_b", combine=True))
posterior_samples_l2w = np.squeeze(trace.get_values("l2_w", combine=True))

# print(posterior_samples_l1w.shape)
# posterior_samples_obj = approx.sample(10000)
# posterior_samples_1 = np.squeeze(posterior_samples_obj.posterior["l1_w"].values)
# posterior_samples_2 = np.squeeze(posterior_samples_obj.posterior["l1_b"].values)
# posterior_samples_3 = np.squeeze(posterior_samples_obj.posterior["l2_w"].values)


### Plot fi (prediction by NN)
print('Max x for training: ',np.max(x_list[0]))
print('Min x for training: ',np.min(x_list[0]))
# expscale = 1*(np.max(ytrain_hat) - np.min(ytrain_hat))
# fi_input = np.expand_dims(np.arange(0,0.5,0.01),axis=1)
fi_input = np.expand_dims(np.arange(np.min(x_list[0]),np.max(x_list[0]),0.01),axis=1)

f_predlist = []

plt.clf()
for i in range(PosteriorSample*4):
    if i%500==0:
        print('fi prediction; Sample:',i)
    
    layer1 = np.tanh(np.dot(fi_input,posterior_samples_l1w[i:(i+1),:]) + posterior_samples_l1b[i:(i+1),:])
    fi_output = np.dot(layer1,posterior_samples_l2w[i,:])
    f_predlist.append(np.squeeze(fi_output))
    plt.plot(np.squeeze(fi_input),np.squeeze(fi_output))
plt.savefig('fi_prediciton.png',bbox_inches='tight')




f_prediction_mean = np.mean(np.asarray(f_predlist),axis=0)
f_prediction_std = np.std(np.asarray(f_predlist),axis=0)


legend_switch = 0
plt.clf()
plt.figure(figsize=(5, 4))
params = {
            'axes.labelsize': 21,
            'font.size': 21,
            'legend.fontsize': 23,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
plt.rcParams.update(params)

if legend_switch == 0:
    plt.plot(fi_input,-np.square(fi_input)+fi_input,'-k',linewidth=3,label='ground truth')
plt.plot(fi_input,f_prediction_mean,'--',color='crimson',linewidth=3,label=r'NN prediction')
plt.fill_between(np.squeeze(fi_input),f_prediction_mean+f_prediction_std,f_prediction_mean-f_prediction_std,color='crimson',alpha=0.4,label='NN uncertainty')
# for i in range(num_ic):
#     plt.scatter(ytrain_hat_list[i],d_hat_list[i],marker='X',s=80,color='crimson',edgecolors='k',label='training data',zorder=2)
if legend_switch == 1:
    plt.plot(fi_input,-np.square(fi_input)+fi_input,'-k',linewidth=3,label='ground truth')
    plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=2,frameon=False)
plt.grid(alpha=0.5)
plt.savefig('result/figure/fi_N'+str(int(NoisePer*100))+'D'+str(int(NumDataTrain))+'.png',bbox_inches='tight')




xlist_array = []
plt.clf()
### dynamics prediction with posterior
# for i in range(1):
for i in range(PosteriorSample*4):
    if i%500==0:
        print('dynamics prediction; Sample:',i)
    ### other model parameters
    x_t0 = 0.01
    dt = 1e-3
    T = 9

    num_T = int(T/dt)
    pre_x = x_t0
    xlist = [x_t0]

    for timestep in range(num_T):
        
        layer1 = np.tanh(np.dot(pre_x,posterior_samples_l1w[i:(i+1),:]) + posterior_samples_l1b[i:(i+1),:])
        f_pred = np.dot(layer1,posterior_samples_l2w[i,:])
        if np.squeeze(f_pred)<0:
            print(pre_x,f_pred)
        next_x = dt*f_pred + pre_x

        xlist.append(next_x.item())
        pre_x = next_x
    
    xlist_array.append(xlist)
    plt.plot(timedata,np.asarray(xlist))
plt.savefig('x_prediciton.png',bbox_inches='tight')

prediction_mean = np.mean(np.asarray(xlist_array),axis=0)
prediction_std = np.std(np.asarray(xlist_array),axis=0)

plt.clf()
plt.figure(figsize=(5, 4))
params = {
            'axes.labelsize': 21,
            'font.size': 21,
            'legend.fontsize': 23,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
plt.rcParams.update(params)

if legend_switch == 0:
    plt.plot(timedata,x_list[0],'-k',linewidth=3,label='ground truth')
plt.plot(timedata,prediction_mean,'--',color='tab:blue',linewidth=3,label=r'$x$ prediction')
plt.fill_between(timedata,prediction_mean+prediction_std,prediction_mean-prediction_std,color='tab:blue',alpha=0.4,label=r'$x$ uncertainty')
# for i in range(num_ic):
#     if i == 0:
#         plt.scatter(Xtrain_list[i],ytrain_list[i],marker='X',s=80,color='tab:blue',edgecolors='k',label='training data',zorder=2)
#     else:
#         plt.scatter(Xtrain_list[i],ytrain_list[i],marker='X',s=80,color='tab:blue',edgecolors='k',zorder=2)
if legend_switch == 1:
    plt.plot(timedata,x_list[0],'-k',linewidth=3,label='ground truth')
    plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=2,frameon=False)

# plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
plt.grid(alpha=0.5)

plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(NumDataTrain))+'.png',bbox_inches='tight')








###################################################################

# para_mean.append(mu_mean)
# para_cova.append(mu_covariance)

# print('Parameter mean:', para_mean)
# print('Parameter covariance: ',para_cova)

# np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
# np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

