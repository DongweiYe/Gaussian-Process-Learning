import numpy as np
import GPy
import matplotlib.pyplot as plt
from dynamical_system import *
from neuralODE import *
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal

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
DataSparsity = 0.025       ### Note different from scenarios 1 and 2, here we take 100% of as total data we have
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0         ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
PosteriorSample = 200    ### posterior sampling numbers
Bayesian = 1            ### 0 -> MAP, 1-> Bayesian
consistent = 0          ### Whether to use consistent structure (this is only for post-processing)
                        ### for neural network, you need to comment out/in the corresponding one

### NN parameters
n_epochs = 15000
posterior_sample_num = 1000
### Load data and add noise
x = np.load('data/sin_x_ic.npy')
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
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*1e-4
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-4
    #print(np.linalg.cond(Kuu))
else:
    Kuu = xtkernel.K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    
    Kdd = xtkernel.dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise ### Additional noise to make sure invertable

Kdu = xtkernel.dK_dX(Xtrain,Xtrain,0)
Kud = Kdu.T
invKuu = np.linalg.inv(Kuu)                                                  

invRdd = Kdd-Kdu@invKuu@Kud
Rdd = np.linalg.inv(invRdd)
print(invRdd.shape)
### Compute the true value of d_i using GP
d_hat = Kdu@invKuu@ytrain

##############################################
############# Build neural ODE ###############
##############################################
if Bayesian == 1:
    NNmodel = BayesNeuralODE(torch.from_numpy(invRdd))

    guide = AutoDiagonalNormal(NNmodel)
    adam = pyro.optim.Adam({"lr": 1e-2})
    svi = SVI(NNmodel, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    bar = trange(n_epochs)

    for epoch in bar:
        loss = svi.step(torch.from_numpy(ytrain_hat).float(), torch.squeeze(torch.from_numpy(d_hat)).float())
        bar.set_postfix(loss=f'{loss / x.shape[0]:.3f}')
    
    guide.requires_grad_(False)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    parameter_mean = pyro.get_param_store()._params['AutoDiagonalNormal.loc']
    parameter_std = torch.exp(pyro.get_param_store()._params['AutoDiagonalNormal.scale'])
    parameter_cov = torch.diag(parameter_std)
    
    distrib = MultivariateNormal(loc=parameter_mean, covariance_matrix=parameter_cov)
    samples = distrib.sample((posterior_sample_num,))
    print(samples.shape)

    xlist_array = []

    if consistent == 1:
        for i in range(samples.shape[0]):

            ### LV other parameters
            x_t0 = 0
            dt = 1e-3
            T = 5

            xlist = sin_model(x_t0,T,dt,[samples[i,0],samples[i,1],samples[i,2]])
            
            xlist_array.append(xlist)
    else:
        ### Plot fi (prediction by NN)
        print('Max x for training: ',np.max(ytrain_hat))
        print('Min x for training: ',np.min(ytrain_hat))
        expscale = 0.5*(np.max(ytrain_hat) - np.min(ytrain_hat))
        fi_input = np.expand_dims(np.arange(np.min(ytrain_hat)-expscale,np.max(ytrain_hat)+expscale,0.01),axis=1)

        f_predlist = []
        for i in range(samples.shape[0]):
            if i%100==0:
                print('fi prediction; Sample:',i)
            layerone = torch.mm(torch.from_numpy(fi_input).float(),samples[i,0:10].view(1,-1))+samples[i,10:20]

            f_pred = torch.mm(torch.tanh(layerone),samples[i,20:30].view(-1, 1))
            f_predlist.append(np.squeeze(f_pred.detach().numpy()))

        f_prediction_mean = np.mean(np.asarray(f_predlist),axis=0)
        f_prediction_std = np.std(np.asarray(f_predlist),axis=0)

        plt.figure(figsize=(5, 4))
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

        
        plt.plot(fi_input,2*np.sin(0.8*fi_input+0.5),'-k',linewidth=3,label='ground truth')
        plt.plot(fi_input,f_prediction_mean,'--',color='crimson',linewidth=3,label=r'NN prediction')
        plt.fill_between(np.squeeze(fi_input),f_prediction_mean+f_prediction_std,f_prediction_mean-f_prediction_std,color='crimson',alpha=0.4,label=r'uncertainty')

        plt.scatter(ytrain_hat,d_hat,marker='X',s=80,color='crimson',edgecolors='k',label='training data',zorder=2)
        

        plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=4,frameon=False)
        # plt.show()
        plt.savefig('result/figure/fi_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')



        ### dynamics prediction with posterior
        for i in range(samples.shape[0]):
            if i%100==0:
                print('dynamics prediction; Sample:',i)
            ### other model parameters
            x_t0 = 0
            dt = 1e-3
            T = 5

            num_T = int(T/dt)
            pre_x = x_t0
            xlist = [x_t0]

            # xlist = sin_model(x_t0,T,dt,[samples[i,0],samples[i,1],samples[i,2]])
            for timestep in range(num_T):
                # pre_x_tensor = torch.tensor([[pre_x]]).to(torch.float64)
                f_pred = torch.mm(samples[i,20:30].view(1, -1),torch.tanh(samples[i,0:10]*pre_x+samples[i,10:20]).view(-1, 1))
                next_x = dt*(f_pred[0][0].detach().numpy()) + pre_x

                xlist.append(next_x)
                pre_x = next_x
            
            xlist_array.append(xlist)


    prediction_mean = np.mean(np.asarray(xlist_array),axis=0)
    prediction_std = np.std(np.asarray(xlist_array),axis=0)


    plt.figure(figsize=(5, 4))
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
    plt.plot(timedata,prediction_mean,'--',color='crimson',linewidth=3,label=r'$x$ prediction')
    plt.fill_between(timedata,prediction_mean+prediction_std,prediction_mean-prediction_std,color='crimson',alpha=0.4,label=r'$x$ uncertainty')

    plt.scatter(Xtrain,ytrain,marker='X',s=80,color='crimson',edgecolors='k',label='training data',zorder=2)

    plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
    
    # plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=4,frameon=False)
    # plt.show()
    plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')



else:
    NNmodel = neuralODE()
    MSEloss = MSERddloss()
    # MSEloss = nn.MSELoss() ### least square (when Rdd -> I)
    optimizer = optim.Adam(NNmodel.parameters(), lr=0.01)
    # optimizer = optim.AdamW(NNmodel.parameters())
    
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
    x_t0 = 0
    dt = 1e-3
    T = 5

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

