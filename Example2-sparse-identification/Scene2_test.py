import numpy as np
import GPy
import pickle
import matplotlib.pyplot as plt
import pymc as pm
from scipy.optimize import minimize
from sklearn.linear_model import Lasso,Ridge
import statsmodels.api as sm
import pytensor as pyte

from LotkaVolterra_model import *
from mcmc import *
from visualization import *
from validation import *

np.random.seed(0)
np.set_printoptions(precision=4)

######################################
############## Scenario 1 ############
######################################
### The covariance matrix is highly possible to singular if no noise is added
### This applies to Kdd, Kuu, Kud and Kdu

### Parameters
TrainRatio = 0.4         ### Train/Test data split ratio
DataSparsity = 0.025      ### Take 25% of as the total data; DataSparsity =0.25 (100% data),DataSparsity=0.025 (10% data),etc
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0          ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
NumDyn = 2               ### number of dynamics equation
IC_test = 0               ### redundant function
prior_name = 'laplace'   ### laplace or spike-slab

spike_std = 1e-2    ### std of spike (normal) distribution
slab_std = 1e0     ### std of slab (normal) distribution


### Load data and add noise
x1 = np.load('data/x1.npy')
x2 = np.load('data/x2.npy')
timedata = np.load('data/time.npy')

NoiseSTD1 = NoisePer*np.mean(x1)
NoiseSTD2 = NoisePer*np.mean(x2)

preydata = x1 + np.random.normal(NoiseMean,NoiseSTD1,x1.shape[0])
preddata = x2 + np.random.normal(NoiseMean,NoiseSTD2,x2.shape[0])

num_data = preddata.shape[0] - 1 ### 0 -> K, using [0,K-1] for int
num_train = int((num_data*TrainRatio)*DataSparsity) 

samplelist = np.random.choice(np.arange(0,int(num_data*TrainRatio)),num_train,replace=False)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(preydata[samplelist],axis=1),np.expand_dims(preddata[samplelist],axis=1)))
ytrain_backup = np.hstack((np.expand_dims(x1[samplelist],axis=1),np.expand_dims(x2[samplelist],axis=1)))

# plt.figure(figsize=(7, 5))
# params = {
#         'axes.labelsize': 21,
#         'font.size': 21,
#         'legend.fontsize': 23,
#         'xtick.labelsize': 21,
#         'ytick.labelsize': 21,
#         'text.usetex': False,
#         'axes.linewidth': 1,
#         'xtick.major.width': 1,
#         'ytick.major.width': 1,
#         'xtick.major.size': 1,
#         'ytick.major.size': 1,
#     }
# plt.rcParams.update(params)
# plt.plot(timedata[:8000],x1[:8000],'-k',linewidth=1.5,label='dynamics')
# plt.plot(Xtrain,ytrain[:,0],'*',color='tab:red',markersize=8,label='data')
# # plt.plot(Xtrain,ytrain[:,1],'*',label='x2 dynamics')

# # plt.plot(timedata,x2,'-k',label='x2')
# plt.xlabel(r'$t$')
# plt.ylabel(r'$x_i(t)$')
# plt.legend(fontsize=12)
# # plt.show()
# plt.savefig('data.png',bbox_inches='tight')

### Build a GP to infer the hyperparameters for each dynamic equation 
### and proper G_i data from regression
ytrain_hat = []
kernellist = []
GPlist = []

for i in range(0,NumDyn):
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2)
    # xtkernel = GPy.kern.PeriodicMatern52(input_dim=1, variance=2,period=3,lengthscale=3)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)

    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)
    
    ypred,yvar = xtGP.predict(Xtrain)

    # if i == 0:
    #     plt.plot(timedata[:8000],x1[:8000],'-',color='black',label='GT')
    # else:
    #     plt.plot(timedata[:8000],x2[:8000],'-',color='black',label='GT')
    # plt.plot(Xtrain,ypred,'o',color='tab:red',label='prediction')
    # plt.plot(Xtrain,ytrain[:,i:(i+1)],'*',label='data')
    # plt.legend()
    # plt.savefig('GP_'+str(i)+'.png')
    # plt.clf()
    
    ytrain_hat.append(ypred)

    kernellist.append(xtkernel)
    GPlist.append(xtGP)
    
ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T

para_mean = []
para_cova = []

for i in range(0,NumDyn):
    
    print('Estimate parameters in equation: ',i)

    ### Compute hyperparameters from a GP of x(t)
    GPvariance = kernellist[i][0]
    GPlengthscale = kernellist[i][1]
    GPnoise = GPlist[i]['Gaussian_noise'][0][0]
    # print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

    ### Construct the covariance matrix of equation (5)
    if NoisePer == 0:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*1e-4
    else:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    

    ### smallest Kdd_noise to make Kdd invertable
    Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) 
    Kdd_noise = 1e-5
    print('Condition number of Kdd is:',np.linalg.cond(Kdd))
    print('Determinant of Kdd is:',np.linalg.det(Kdd))
    if np.linalg.det(Kdd) == 0:
        Kdd_updated = Kdd
        while np.linalg.det(Kdd_updated) == 0:
            print('Kdd determinant too small, adding noise:', Kdd_noise)
            Kdd_updated = Kdd + np.identity(Xtrain.shape[0])*Kdd_noise
            print('New condition number:',np.linalg.cond(Kdd_updated))
            print('New determinant:',np.linalg.det(Kdd_updated))
            Kdd_noise = Kdd_noise*10
        Kdd = Kdd_updated
    
    Kdu = kernellist[i].dK_dX(Xtrain,Xtrain,0)                                                  
    Kud = Kdu.T                                                                                 
    invKuu = np.linalg.inv(Kuu)

    invRdd = Kdd-Kdu@invKuu@Kud
    Rdd = np.linalg.inv(invRdd)
    Rdu = -Rdd@Kdu@invKuu
    Rud = Rdu.T

    d_hat = Kdu@invKuu@ytrain[:,i:(i+1)]

    ### Construct dictionary G for sindy regression (currently deterministic version)
    Gdata = np.hstack((
                    np.ones((ytrain_hat[:,0:1].shape[0],1)),   \
                    ytrain_hat[:,0:1],                          \
                    ytrain_hat[:,1:2],                          \
                    np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,0:1]),  \
                    np.multiply(ytrain_hat[:,1:2],ytrain_hat[:,1:2]),  \
                    np.multiply(ytrain_hat[:,0:1],ytrain_hat[:,1:2]))) 

    
    ### Threshold truncation using ridge regression
    def ridge_reg(g_data,d_data,regu):
        regressor = Ridge(alpha = regu)
        regressor.fit(g_data,d_data)
        # print('Regu parameter:',regressor.coef_)
        return regressor.coef_

    def regulized_GLS(g_data,d_data,cov_matrix):
        gls_model = sm.GLS(d_data, g_data, sigma=cov_matrix)
        gls_results = gls_model.fit_regularized(method='elastic_net',alpha=1,L1_wt=0)
        # gls_results = gls_model.fit()
        return gls_results.params

    threshold = 0.2
    # theta_test = ridge_reg(Gdata,d_hat,1)
    theta_test = regulized_GLS(Gdata,d_hat,invRdd).reshape(1,-1)
    print('L2 esimation of parameters:', theta_test)

    Gdata_sparsity = Gdata
    truncation_index = 0
    truncation_index_new = 1
    while np.min(np.abs(theta_test)) < threshold and np.sum(truncation_index)-np.sum(truncation_index_new)!=0:
        truncation_index = np.where(np.abs(theta_test) < threshold)[1]
        print('Truncation index: ',truncation_index)
        # Gdata_sparsity = np.delete(Gdata_sparsity, truncation_index,1) #[:,truncation_index] = 0
        Gdata_sparsity[:,truncation_index] = 0
        theta_test = ridge_reg(Gdata_sparsity,d_hat,1)
        print('L2 esimation of parameters:', theta_test)
        truncation_index_new = np.where(np.abs(theta_test) < threshold)[1]

    sparsity_index = np.squeeze(theta_test)
    sparsity_index[sparsity_index!=0] = 1
    print('Final sparsity index: ', sparsity_index)
    ### Step 1, find for a proper start of lambda (lasso regression without Rdd norm)
    # lambda_trial = np.linspace(0, 0.1, num=100,endpoint=False)
    
    # def LASSO_reg(regu_para):
    #     if isinstance(regu_para, int) == True or isinstance(regu_para, float) == True:
    #         lasso_regressor = Lasso(alpha = regu_para)
    #     else:
    #         lasso_regressor = Lasso(alpha = regu_para[0])
    #     lasso_regressor.fit(Gdata,d_hat)
    #     MAP_value = np.sum(np.square(Gdata@lasso_regressor.coef_ - np.squeeze(d_hat))) + 2*Kuu.shape[0]*lambda_trial[lambda_i]* np.sum(np.abs(lasso_regressor.coef_))
    #     print('Regu parameter:',regu_para,'; MAP:',MAP_value,'; Coefficients:',lasso_regressor.coef_)
    #     return MAP_value

    # MAParray = []
    # for lambda_i in range(len(lambda_trial)):
    #     MAParray.append(LASSO_reg(lambda_trial[lambda_i])) 
    # min_index = np.argmin(np.asarray(MAParray))
    # init_guess = lambda_trial[min_index]*Kuu.shape[0]
    # print('Index:',min_index )
    # print('Select initial guess:', init_guess)
    # if init_guess == 0:
    #     init_guess = 1e-8
    ### Optimise with MAP for the best lambda
    # result = minimize(LASSO_reg, init_guess, method="nelder-mead")

    basic_model = pm.Model()

    with basic_model:
        # Priors for theta, parameter b in laplace is actually 1/lambda (lambda is penalty coef) 
        if prior_name == 'laplace':
            b_sparse = 1e-8
            b_nonsparse = 1e1
            b_list = []
            for cand_i in range(6):
                if sparsity_index[cand_i] == 0:
                    b_list.append([b_sparse])
                else:
                    b_list.append([b_nonsparse])
            print(b_list)
            theta = pm.Laplace("theta", mu=0, b=pyte.tensor.stack(b_list),shape=(6,1))
            # Kddnoise = pm.Normal("Kddnoise",mu=0, sigma=1e2)

        elif prior_name == 'spike-slab':
            bi_sample = pm.Bernoulli('bi_sample',p=0.5,shape=(6,1))
            spike = pm.Laplace("spike", mu=0, b=spike_std,shape=(6,1))
            slab = pm.Normal("slab", mu=0, sigma=slab_std,shape=(6,1))
            theta = slab*bi_sample + spike*(1-bi_sample)
        
        covariance = invRdd
        # covariance = np.identity(invRdd.shape[0])

        mu = Gdata@theta
        Y_obs = pm.MvNormal('Y_obs', mu=mu, cov=covariance, observed=d_hat)
        approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)]) 
        # trace = pm.sample(500, return_inferencedata=False,cores=4,tune=500,random_seed=0)

    if prior_name == 'laplace':
        # posterior_samples = np.squeeze(trace.get_values('theta', combine=True))
        # sindy_dist(posterior_samples,str(i))
        
        posterior_samples_obj = approx.sample(1000)
        posterior_samples = np.squeeze(posterior_samples_obj.posterior["theta"].values)
        print(np.mean(posterior_samples,axis=0))
        print(np.std(posterior_samples,axis=0))
        multiplot_dist(posterior_samples,str(i),str(i)+'_D'+str(int(DataSparsity*400))+'_N'+str(int(NoisePer*100)))
        print(posterior_samples.shape)
        
        
        # print('Active term:',(np.abs(np.mean(posterior_samples,axis=0)) > 0.3).sum())
        # ### Not sparse, skip this round
        # if (np.abs(np.mean(posterior_samples,axis=0)) > 0.3).sum() >=4:
        #     print('No sparsity, break')
        #     break
        # ### Too sparse skip this round
        # elif np.sum(np.abs(np.mean(posterior_samples,axis=0))) <2e-1: 
        #     print('Too much sparsity, break')
        #     break
        # else:
        #     para_mean.append(np.mean(posterior_samples,axis=0))
        #     para_cova.append(np.std(posterior_samples,axis=0))


    elif prior_name == 'spike-slab':
        posterior_samples_bi = np.squeeze(trace.get_values('bi_sample', combine=True))
        posterior_samples_spike = np.squeeze(trace.get_values('spike', combine=True))
        posterior_samples_slab = np.squeeze(trace.get_values('slab', combine=True))

        print(posterior_samples_bi)
        print(np.mean(posterior_samples_bi,axis=0))
        print(np.mean(posterior_samples_slab,axis=0))
        print(np.mean(posterior_samples_spike,axis=0))
    
        print(np.mean(posterior_samples_bi,axis=0)*np.mean(posterior_samples_slab,axis=0) + (1-np.mean(posterior_samples_bi,axis=0))* np.mean(posterior_samples_spike,axis=0))
        # print(np.std(posterior_samples_bi,axis=0),np.std(posterior_samples_spike,axis=0),np.std(posterior_samples_slab,axis=0))

print('Inferred parameter:',para_mean)
print('Inferred parameter (std):',para_cova)
### Check inferred parameter is sparse or too sparse


# valid_error = pred_validation(regu_para,x1,x2,Xtrain,ytrain,para_mean)
# print('Validate error:',valid_error)


# error_array = []
# for b_i in range(laplace_b.shape[0]):
#     error_array.append(gpndy(laplace_b[b_i,:]))

# min_error = np.argmin(error_array)
# print(error_array)
# print(min_error)

# x0_guess = laplace_b[min_error-1,:]
# # bnds = ((-4,-1),(-4,-1))
# result = minimize(gpndy, x0_guess, method="nelder-mead",options={'adaptive': True})


     

# np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
# np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

# ### Prediction with marginalization
# preylist_array = []
# predlist_array = []

# for i in range(PosteriorSample):

#     mu1 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[0]),para_cova[0],1))
#     mu2 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[1]),para_cova[1],1))

#     ### LV other parameters
#     if IC_test == 0:
#         x1_t0 = 1
#         x2_t0 = 1
#         T = 20
#     else:
#         x1_t0 = 2
#         x2_t0 = 1.2
#         T = 10

#     dt = 1e-3

#     preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[mu1[0],-mu1[1],mu2[0],-mu2[1]])
#     if IC_test == 0:
#         if np.max(preylist) > 20 or np.max(predatorlist) > 20:
#             pass
#         else:
#             preylist_array.append(preylist)
#             predlist_array.append(predatorlist)
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

#     plt.fill_between(new_timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5,label=r'$x_1$ uncertainty')
#     plt.fill_between(new_timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5,label=r'$x_2$ uncertainty')
#     plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
#     plt.show()
#     # plt.savefig('result/figure/ICs/'+str(x1_t0)+'&'+str(x2_t0)+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')

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
#     plt.show()
#     # plt.savefig('result/figure/1N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')
    