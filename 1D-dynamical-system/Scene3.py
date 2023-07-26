import numpy as np
import GPy
import matplotlib.pyplot as plt
from dynamical_system import *
from neuralODE import *
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm

np.random.seed(0)
torch.manual_seed(0)
pyro.set_rng_seed(0)



########################################
############## Scenario 2 ##############
########################################

### Parameters
TrainRatio = 0.25         ### Train/Test data split ratio
DataSparsity = 0.25       ### Note different from scenarios 1 and 2, here we take 100% of as total data we have
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0.1          ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
PosteriorSample = 200    ### posterior sampling numbers
Bayesian = 1

### NN parameters
n_epochs = 5000

### Load data and add noise
x = np.load('data/tanh_x_ic1.npy')
timedata = np.load('data/time.npy')

NoiseSTD = NoisePer*np.mean(x)
xdata = x + np.random.normal(NoiseMean,NoiseSTD,x.shape[0])

### Compute the required training data and randomly sample from the list
num_data = x.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 
samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)

print('Data from training: ',num_train)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.expand_dims(xdata[samplelist],axis=1)

### Visualization
plt.plot(timedata,x,'*',label='x dynamics')
plt.legend()
plt.show()

### Build a GP to infer the hyperparameters for each dynamic equation 
### and compute d_i conditioning on u_i
xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
xtGP = GPy.models.GPRegression(Xtrain,ytrain,xtkernel)

xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
xtGP.optimize_restarts(num_restarts=2,verbose=False)

ytrain_hat,ytrain_hat_var = xtGP.predict(Xtrain)

# ### Vsiualization
plt.plot(Xtrain,ytrain,'b*',label='GT')
plt.plot(Xtrain,ytrain_hat,'o',color='tab:red',label='prediction',alpha=0.5)
plt.legend()
plt.show()

### Print out hyperparameters from a GP of x(t)
GPvariance = xtkernel[0]
GPlengthscale = xtkernel[1]
GPnoise = xtGP['Gaussian_noise'][0][0]
print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

### Construct the covariance matrix 
if NoisePer == 0:
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*1e-6
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-6
    #print(np.linalg.cond(Kuu))
else:
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise ### Additional noise to make sure invertable

Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)
Kud = Kdu.T
invKuu = np.linalg.inv(Kuu)                                                  

invRdd = Kdd-Kdu@invKuu@Kud
Rdd = np.linalg.inv(invRdd)

### Compute the true value of d_i using GP
d_hat = Kdu@invKuu@ytrain

##############################################
############# Build neural ODE ###############
##############################################
if Bayesian == 1:
    NNmodel = BayesNeuralODE(torch.from_numpy(invRdd))
    # nuts_kernel = NUTS(NNmodel, jit_compile=False)
    # mcmc = MCMC(nuts_kernel, num_samples=500)
    # mcmc.run(torch.from_numpy(ytrain_hat), torch.from_numpy(d_hat))
    # predictive = Predictive(model=NNmodel, posterior_samples=mcmc.get_samples())
    # preds = predictive(torch.from_numpy(ytrain_hat))


    guide = AutoDiagonalNormal(NNmodel)
    adam = pyro.optim.Adam({"lr": 1e-3})
    svi = SVI(NNmodel, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    bar = trange(10000)

    for epoch in bar:
        loss = svi.step(torch.from_numpy(ytrain_hat), torch.squeeze(torch.from_numpy(d_hat)))
        bar.set_postfix(loss=f'{loss / x.shape[0]:.3f}')

    # print(torch.squeeze(NNmodel.fc1_weight).mean(),torch.squeeze(NNmodel.fc1_bias).mean(),torch.squeeze(NNmodel.fc2_weight).mean())
    # print(torch.squeeze(NNmodel.fc1_weight).std(),torch.squeeze(NNmodel.fc1_bias).std(),torch.squeeze(NNmodel.fc2_weight).std())
    
    predictive = Predictive(model=NNmodel, guide=guide, num_samples=500)
    preds = predictive(torch.from_numpy(ytrain_hat),None)
    
    for k, v in preds.items():
        print(f"{k}: {tuple(v.shape)}")

    a1 = preds['fc1_weight'].T.detach().numpy().mean()
    a2 = preds['fc1_bias'].T.detach().numpy().mean()
    a3 = preds['fc2_weight'].T.detach().numpy().mean()

    print('parameters:',a1,a2,a3)

    # y_mean = np.mean(np.mean(ypreds,axis=2),axis=1)
    # y_std = np.std(np.std(ypreds,axis=2),axis=1)
  

    # plt.plot(ytrain_hat,d_hat,'or',label='d_i')
    # plt.plot(ytrain_hat, y_mean, '*', linewidth=3, color="#408765", label="predictive mean")
    # # plt.fill_between(np.squeeze(ytrain_hat), np.squeeze(y_mean - 2 * y_std), np.squeeze(y_mean + 2 * y_std), alpha=0.6, color='#86cfac', zorder=5)
    # plt.legend(loc=4, fontsize=15, frameon=False)
    # plt.show()



else:
    NNmodel = neuralODE()
    MSEloss = MSERddloss() ### 
    # MSEloss = nn.MSELoss() ### least square (when Rdd -> I)
    optimizer = optim.Adam(NNmodel.parameters(), lr=0.01)
    # dataset = TensorDataset(torch.from_numpy(ytrain_hat), torch.from_numpy(d_hat))
    # dataloader = DataLoader(dataset, batch_size=num_train, shuffle=False)

    for i in range(n_epochs):
        
        f_pred = NNmodel(torch.from_numpy(ytrain_hat))
        cost = MSEloss(f_pred,torch.from_numpy(d_hat),torch.from_numpy(Rdd))

        #backprop
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print loss
        if i%2000 == 0:
            print('Loss:',cost)

    ### Print out the parameters
    print("Model Parameters:")
    for name, param in NNmodel.named_parameters():
        print(param)

    ### Visualization prediction of f(x) and compare it to di
    prediction = NNmodel(torch.from_numpy(ytrain_hat))
    plt.plot(ytrain_hat,prediction.detach().numpy(),'*k',label='NN prediction')
    plt.plot(ytrain_hat,d_hat,'or',label='d_i')
    plt.legend()
    plt.show()




####################### Temp visualization ########################
para_mean = []
for name, param in NNmodel.named_parameters():
    para_mean.append(param[0].detach().numpy().item())

### Set other parameters and rerun the model 
x_t0 = 1.5
dt = 1e-3
T = 20

# print(para_mean)
xlist = tanh_model(x_t0,T,dt,[para_mean[0],para_mean[1],para_mean[2]])

# num_T = int(T/dt)
# pre_x = x_t0
# xlist = [x_t0]

# for timestep in range(num_T):
#     pre_x_tensor = torch.tensor([[pre_x]]).to(torch.float64)
#     f_pred = NNmodel(pre_x_tensor)
#     next_x = dt*(f_pred[0][0].detach().numpy()) + pre_x

#     xlist.append(next_x)
#     pre_x = next_x

# xlist = np.array(xlist)

plt.figure(figsize=(17, 2))
params = {
            'axes.labelsize': 21,
            'font.size': 21,
            'legend.fontsize': 23,
            'xtick.labelsize': 21,
            'ytick.labelsize': 21,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
plt.rcParams.update(params)
plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')
plt.plot(timedata,xlist,'--',color='tab:orange',linewidth=3,label=r'$x$ prediction')
plt.scatter(Xtrain,ytrain,marker='X',s=80,color='darkorange',edgecolors='k',label='training data',zorder=2)
plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
# plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')

# if NoisePer == 0:
#     plt.ylim([-0.8,8])
# plt.xlim([-1,20])
# plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
plt.show()
# plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')
# plt.savefig('result/figure/legend.png',bbox_inches='tight')




###################################################################

# para_mean.append(mu_mean)
# para_cova.append(mu_covariance)

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