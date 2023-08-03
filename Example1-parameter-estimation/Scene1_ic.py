import numpy as np
import GPy
import pickle
import matplotlib.pyplot as plt
from LotkaVolterra_model import *
# np.random.seed(0)



######################################
############## Scenario 1 ############
######################################
### The covariance matrix is highly possible to singular if no noise is added
### This applies to Kdd, Kuu, Kud and Kdu

### Parameters
TrainRatio = 0.4         ### Train/Test data split ratio
DataSparsity = 0.0025      ### Take 25% of as the total data we have
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0.1           ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
NumDyn = 2               ### number of dynamics equation
PosteriorSample = 1000    ### posterior sampling numbers
num_ic = 1

### Load data and add noise
x1_list = []
x2_list = []
for i in range(num_ic):
    x1_list.append(np.load('data/x1_'+str(i)+'.npy'))
    x2_list.append(np.load('data/x2_'+str(i)+'.npy'))
timedata = np.load('data/time.npy')

NoiseSTD1_list = []
NoiseSTD2_list = []
preydata_list = []
preddata_list = []
ytrain_list = []

num_data = timedata.shape[0] - 1 ### 0 -> K, using [0,K-1] for int ,for each ic
num_train = int((num_data*TrainRatio)*DataSparsity)

### Read in the random state from non-IC case, make sure one set of data is same.
with open('state_time.obj', 'rb') as f:
    np.random.set_state(pickle.load(f))

samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)
Xtrain = np.expand_dims(timedata[samplelist],axis=1)

### Read in the random state from non-IC case, make sure one set of data is same.
with open('state_noise.obj', 'rb') as f:
    np.random.set_state(pickle.load(f))

for i in range(num_ic):
    
    NoiseSTD1_list.append(NoisePer*np.mean(x1_list[i]))
    NoiseSTD2_list.append(NoisePer*np.mean(x2_list[i]))

    print(np.random.normal(NoiseMean,NoiseSTD1_list[i],x1_list[i].shape[0]))

    preydata_list.append(x1_list[i] + np.random.normal(NoiseMean,NoiseSTD1_list[i],x1_list[i].shape[0]))
    preddata_list.append(x2_list[i] + np.random.normal(NoiseMean,NoiseSTD2_list[i],x2_list[i].shape[0]))

    ytrain_list.append(np.hstack((np.expand_dims(preydata_list[i][samplelist],axis=1),np.expand_dims(preddata_list[i][samplelist],axis=1))))

    plt.plot(Xtrain,ytrain_list[i][:,0],'*',label='x1 data')
    plt.plot(Xtrain,ytrain_list[i][:,1],'*',label='x2 data')
    plt.plot(timedata,x1_list[i],'-k',label='x1')
    plt.plot(timedata,x2_list[i],'-k',label='x2')
    plt.legend()
    plt.show()

# print(ytrain_list)
### Build a GP to infer the hyperparameters for each dynamic equation 
### and proper G_i data from regression (with 5 ic each)

ytrain_hat_list = []
kernel_list = []
GP_list = []


for i in range(0,NumDyn):
    ytrain_hat_ic = []
    kernel_ic = []
    GP_ic = []
    for j in range(num_ic):
        xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
        xtGP = GPy.models.GPRegression(Xtrain,ytrain_list[j][:,i:(i+1)],xtkernel)

        xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
        xtGP.optimize_restarts(num_restarts=5,verbose=False)
        
        ypred,yvar = xtGP.predict(Xtrain)

        plt.plot(Xtrain,ypred,'*',label='prediction')
        plt.plot(Xtrain,ytrain_list[j][:,i:(i+1)],'*',label='GT')
        plt.legend()
        plt.show()
        
        ytrain_hat_ic.append(ypred)
        kernel_ic.append(xtkernel)
        GP_ic.append(xtGP)
    ytrain_hat_list.append(ytrain_hat_ic)
    kernel_list.append(kernel_ic)
    GP_list.append(GP_ic)


### Need to move this to somewhere below !!!!!!!!!!!!!!
# ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T

## loop for the estimation of each dynamic equation
para_mean = []
para_cova = []


for i in range(0,NumDyn):
    print('Estimate parameters in equation: ',i)
    Kuu_list = []
    Kdd_list = []
    Kdu_list = []
    Kud_list = []
    Rdd_list = []
    Rdu_list = []
    invKuu_list = []
    G_list = []
    for j in range(num_ic):
        print('     Prepare IC data: ',j)

        ### Compute hyperparameters from a GP of x(t)
        GPvariance = kernel_list[i][j][0]
        GPlengthscale = kernel_list[i][j][1]
        GPnoise = GP_list[i][j]['Gaussian_noise'][0][0]
        print('     GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

        ### Construct the covariance matrix of equation (5)
        if NoisePer == 0:
            Kuu_list.append(kernel_list[i][j].K(Xtrain) + np.identity(Xtrain.shape[0])*1e-4)
            Kdd_list.append(kernel_list[i][j].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-4)
        else:
            Kuu_list.append(kernel_list[i][j].K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise)                    ### invertable
            Kdd_list.append(kernel_list[i][j].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise) ### Additional noise to make sure invertable

        Kdu_list.append(kernel_list[i][j].dK_dX(Xtrain,Xtrain,0))                                                  ### not invertable
        Kud_list.append(Kdu_list[j].T)                                                                                 ### not invertable
        invKuu_list.append(np.linalg.inv(Kuu_list[j]))                                                  

        ### If we could only assume that Kuu is invertalbe, then
        Rdd_list.append(np.linalg.inv(Kdd_list[j]-Kdu_list[j]@invKuu_list[j]@Kud_list[j]))
        Rdu_list.append(-Rdd_list[j]@Kdu_list[j]@invKuu_list[j])

        if i == 0:
            G_list.append(np.hstack((ytrain_hat_list[0][j],np.multiply(ytrain_hat_list[0][j],ytrain_hat_list[1][j]))))
        else:
            G_list.append(np.hstack((np.multiply(ytrain_hat_list[0][j],ytrain_hat_list[1][j]),ytrain_hat_list[1][j])))
    
    for j in range(num_ic):
        if j == 0:
            mu_covariance_inv = G_list[j].T@Rdd_list[j]@G_list[j]
            mu_addition = G_list[j].T@Rdu_list[j]@ytrain_list[j][:,i:(i+1)]
        else:
            mu_covariance_inv = mu_covariance_inv + G_list[j].T@Rdd_list[j]@G_list[j]
            mu_addition = mu_addition + G_list[j].T@Rdu_list[j]@ytrain_list[j][:,i:(i+1)]
    
    mu_covariance = np.linalg.inv(mu_covariance_inv)
    mu_mean = - mu_covariance@mu_addition

    para_mean.append(mu_mean)
    para_cova.append(mu_covariance)

print('Parameter mean:', para_mean)
print('Parameter covariance: ',para_cova)

# # np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
# # np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

# ### Prediction with marginalization
# preylist_array = []
# predlist_array = []

# for i in range(PosteriorSample):

#     mu1 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[0]),para_cova[0],1))
#     mu2 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[1]),para_cova[1],1))

#     ### LV other parameters
#     x1_t0 = 2.5
#     x2_t0 = 3

#     dt = 1e-3
#     T = 20

#     preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[mu1[0],-mu1[1],mu2[0],-mu2[1]])
#     if np.max(preylist) > 20 or np.max(predatorlist) > 20:
#         pass
#     else:
#         preylist_array.append(preylist)
#         predlist_array.append(predatorlist)


# #### Addition result for different initial conditions
# if IC_test == 1:
#     preylist_IC,predatorlist_IC = LVmodel(x1_t0,x2_t0,T,dt,[1.5,1,1,3])



# preymean = np.mean(np.asarray(preylist_array),axis=0)
# predmean = np.mean(np.asarray(predlist_array),axis=0)
# preystd = np.std(np.asarray(preylist_array),axis=0)
# predstd = np.std(np.asarray(predlist_array),axis=0)




# if IC_test == 1:
#     plt.figure(figsize=(9, 2))
#     params = {
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
#     plt.rcParams.update(params)
#     new_timedata = np.arange(0,T+T/(T/dt)*0.1,T/(T/dt))
#     plt.plot(new_timedata,preylist_IC,'-k',linewidth=3,label='ground truth')
#     plt.plot(new_timedata,predatorlist_IC,'-k',linewidth=3)

#     plt.plot(new_timedata,preymean,'--',color='royalblue',linewidth=3,label=r'$x_1$ prediction')
#     plt.plot(new_timedata,predmean,'--',color='tab:orange',linewidth=3,label=r'$x_2$ prediction')

#     plt.fill_between(new_timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5)
#     plt.fill_between(new_timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5)
#     plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
#     # plt.show()
#     plt.savefig('result/figure/ICs/'+str(x1_t0)+'&'+str(x2_t0)+'111.png',bbox_inches='tight')

# else:
#     plt.figure(figsize=(17, 2))
#     params = {
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
#     plt.rcParams.update(params)
    

#     plt.plot(timedata,preymean,'--',color='royalblue',linewidth=3,label=r'$x_1$ prediction')
#     plt.plot(timedata,predmean,'--',color='tab:orange',linewidth=3,label=r'$x_2$ prediction')

#     plt.fill_between(timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5,label=r'$x_1$ uncertainty')
#     plt.fill_between(timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5,label=r'$x_2$ uncertainty')

#     plt.scatter(Xtrain,ytrain[:,0],marker='X',s=80,color='royalblue',edgecolors='k',label='training data '+r'($x_1$)',zorder=2)
#     plt.scatter(Xtrain,ytrain[:,1],marker='X',s=80,color='darkorange',edgecolors='k',label='training data '+r'($x_2$)',zorder=2)

#     plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')

#     plt.plot(timedata,x1,'-k',linewidth=3,label='ground truth')
#     plt.plot(timedata,x2,'-k',linewidth=3)
    

#     if NoisePer == 0:
#         plt.ylim([-0.8,8])
#     # plt.xlim([-1,20])
#     plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=4,frameon=False)
#     # plt.show()
#     plt.savefig('result/figure/1N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')
    