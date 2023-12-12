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
TrainRatio = 0.25         ### Train/Test data split ratio
DataSparsity = 0.025       ### Similar to previous examples, we take 25% of data as total data, i.e. DataSparsity = 0.25 (100% data)
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0         ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
ShapeNN = [1,8,8,1]        ### nodes of each layer of a neural network, e.g. [1,8,8,1]
PosteriorSample = 1000    ### number of posterior samples from MCMC

### Load data and add noise
x = np.load('data/ODEdata.npy')
timedata = np.load('data/time.npy')

NoiseSTD = NoisePer*np.mean(x)
xdata = x + np.random.normal(NoiseMean,NoiseSTD,x.shape[0])

### Compute the required training data and randomly sample from the list
num_data = x.shape[0] - 1  ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)

print('Data from training: ',num_train)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.expand_dims(xdata[samplelist],axis=1)


### Build a GP to infer the hyperparameters for each dynamic equation 
### and compute d_i conditioning on u_i
xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
xtGP = GPy.models.GPRegression(Xtrain,ytrain,xtkernel)

xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
xtGP.optimize_restarts(num_restarts=2,verbose=False)

ytrain_hat,ytrain_hat_var = xtGP.predict(Xtrain)

### Vsiualization
plt.plot(Xtrain,ytrain,'b*',label='GT')
plt.plot(Xtrain,ytrain_hat,'o',color='tab:red',label='GP prediction',alpha=0.5)
plt.legend()
plt.savefig('GP_and_data.png')

### Print out hyperparameters from a GP of x(t)
GPvariance = xtkernel[0]
GPlengthscale = xtkernel[1]
GPnoise = xtGP['Gaussian_noise'][0][0]
print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

### Construct the covariance matrix 
if NoisePer == 0:
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*1e-8
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-8
else:
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise

Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)
Kud = Kdu.T
invKuu = np.linalg.inv(Kuu)                                                  

invRdd = Kdd-Kdu@invKuu@Kud
Rdd = np.linalg.inv(invRdd)

### Compute the true value of d_i using GP
d_hat = Kdu@invKuu@ytrain


##############################################
############# Build neural network ###########
##############################################

### Construct structure of NN to the dynamics systems
with pm.Model() as model:

    ### Weights and bias from input to hidden layer
    layer1_weight = pm.Normal('l1_w', 0, sigma=10, shape=(ShapeNN[0],ShapeNN[1]))
    layer1_bias = pm.Normal('l1_b', 0, sigma=10, shape=ShapeNN[1])
    layer2_weight = pm.Normal('l2_w', 0, sigma=10, shape=(ShapeNN[1],ShapeNN[2]))
    layer2_bias = pm.Normal('l2_b', 0, sigma=10, shape=ShapeNN[2])
    layer3_weight = pm.Normal('l3_w', 0, sigma=10, shape=(ShapeNN[2],ShapeNN[3]))

    ### Feedforward
    layer1 = pm.math.tanh(pm.math.dot(ytrain_hat,layer1_weight) + layer1_bias)
    layer2 = pm.math.tanh(pm.math.dot(layer1,layer2_weight) + layer2_bias)
    out = pm.math.dot(layer2,layer3_weight)
    # layer1 = pm.Deterministic('a1',pm.math.tanh(layer1_weight*ytrain_hat+layer1_bias))
    # layer2 = pm.Deterministic('a2',layer2_weight*layer1)

    Y_obs = pm.MvNormal('Y_obs', mu=out, cov=invRdd, observed=d_hat)
    
    step = pm.Metropolis()
    trace = pm.sample(PosteriorSample,step=step, return_inferencedata=False,cores=4,tune=1000,random_seed=0)
    # approx = pm.fit(100000,method='fullrank_advi',random_seed=0) ### VI


posterior_samples_l1w = np.squeeze(trace.get_values("l1_w", combine=True))
posterior_samples_l1b = np.squeeze(trace.get_values("l1_b", combine=True))
posterior_samples_l2w = np.squeeze(trace.get_values("l2_w", combine=True))
posterior_samples_l2b = np.squeeze(trace.get_values("l2_b", combine=True))
posterior_samples_l3w = np.squeeze(trace.get_values("l3_w", combine=True))

# posterior_samples_obj = approx.sample(10000)
# posterior_samples_1 = np.squeeze(posterior_samples_obj.posterior["l1_w"].values)
# posterior_samples_2 = np.squeeze(posterior_samples_obj.posterior["l1_b"].values)
# posterior_samples_3 = np.squeeze(posterior_samples_obj.posterior["l2_w"].values)


### Plot fi (prediction by NN)
print('Max x for training: ',np.max(ytrain_hat))
print('Min x for training: ',np.min(ytrain_hat))
expscale = 1*(np.max(ytrain_hat) - np.min(ytrain_hat))
# fi_input = np.expand_dims(np.arange(0,0.5,0.01),axis=1)
fi_input = np.expand_dims(np.arange(np.min(ytrain_hat)-expscale,np.max(ytrain_hat)+expscale,0.01),axis=1)

f_predlist = []
for i in range(PosteriorSample*4):
    if i%500==0:
        print('fi prediction; Sample:',i)
    
    layer1 = np.tanh(np.dot(fi_input,posterior_samples_l1w[i:(i+1),:]) + posterior_samples_l1b[i:(i+1),:])
    layer2 = np.tanh(np.dot(layer1,posterior_samples_l2w[i,:,:]) + posterior_samples_l2b[i:(i+1),:])
    fi_output = np.dot(layer2,posterior_samples_l3w[i,:])
    f_predlist.append(np.squeeze(fi_output))

f_prediction_mean = np.mean(np.asarray(f_predlist),axis=0)
f_prediction_std = np.std(np.asarray(f_predlist),axis=0)


legend_switch = 1
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
plt.scatter(ytrain_hat,d_hat,marker='X',s=80,color='crimson',edgecolors='k',label='training data',zorder=2)
if legend_switch == 1:
    plt.plot(fi_input,-np.square(fi_input)+fi_input,'-k',linewidth=3,label='ground truth')
    plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=2,frameon=False)
plt.grid(alpha=0.5)
plt.savefig('result/figure/fi_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')




xlist_array = []
plt.clf()
### dynamics prediction with posterior
for i in range(PosteriorSample*4):
    if i%500==0:
        print('dynamics prediction; Sample:',i)
    ### other model parameters
    x_t0 = 0.1
    dt = 1e-3
    T = 5

    num_T = int(T/dt)
    pre_x = x_t0
    xlist = [x_t0]

    for timestep in range(num_T):
        
        layer1 = np.tanh(np.dot(pre_x,posterior_samples_l1w[i:(i+1),:]) + posterior_samples_l1b[i:(i+1),:])
        layer2 = np.tanh(np.dot(layer1,posterior_samples_l2w[i,:,:]) + posterior_samples_l2b[i:(i+1),:])
        f_pred = np.dot(np.squeeze(layer2),posterior_samples_l3w[i,:])
        next_x = dt*f_pred + pre_x

        xlist.append(next_x.item())
        pre_x = next_x
    
    xlist_array.append(xlist)

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
    plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')
plt.plot(timedata,prediction_mean,'--',color='tab:blue',linewidth=3,label=r'$x$ prediction')
plt.fill_between(timedata,prediction_mean+prediction_std,prediction_mean-prediction_std,color='tab:blue',alpha=0.4,label=r'$x$ uncertainty')
plt.scatter(Xtrain,ytrain,marker='X',s=80,color='tab:blue',edgecolors='k',label='training data',zorder=2)
if legend_switch == 1:
    plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')
    plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=2,frameon=False)

plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
plt.grid(alpha=0.5)

plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')








###################################################################

# para_mean.append(mu_mean)
# para_cova.append(mu_covariance)

# print('Parameter mean:', para_mean)
# print('Parameter covariance: ',para_cova)

# np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
# np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

