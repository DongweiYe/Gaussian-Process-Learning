import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
from dynamical_system import *
import pymc as pm
from sklearn.preprocessing import MinMaxScaler


np.random.seed(0)
########################################
############## Scenario 2 ##############
########################################

### Parameters
TrainRatio = 1            ### Train/Test data split ratio
NumDataTrain = 500       ### Similar to previous examples, we take 25% of data as total data, i.e. DataSparsity = 0.25 (100% data)
NoiseMean = 0             ### 0 mean for white noise
NoisePer = 0              ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
NumDyn = 2                ### number of dynamics equation
PosteriorSample = 250    ### number of posterior samples from MCMC for each core, total number = PosteriorSample*4
e = 0.5
p = 100

### Load data and add noise
x1 = np.load('data/BBH_x1.npy')
x2 = np.load('data/BBH_x2.npy')
timedata = np.load('data/time.npy')

NoiseSTD1 = NoisePer*np.mean(x1)
NoiseSTD2 = NoisePer*np.mean(x2)

x1data = x1 + np.random.normal(NoiseMean,NoiseSTD1,x1.shape[0])
x2data = x2 + np.random.normal(NoiseMean,NoiseSTD2,x2.shape[0])

### Compute the required training data and randomly sample from the list
total_data = x1data.shape[0] - 1  ### 0 -> K, using [0,K-1] for int
num_train = NumDataTrain
samplelist = np.random.choice(np.arange(0,int(total_data*TrainRatio)),num_train,replace=False)

print('Data from training: ',num_train)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(x1data[samplelist],axis=1),np.expand_dims(x2data[samplelist],axis=1)))
ytrain_backup = np.hstack((np.expand_dims(x1[samplelist],axis=1),np.expand_dims(x2[samplelist],axis=1)))

ytrain_noisefree = np.hstack((np.expand_dims(x1[samplelist],axis=1),np.expand_dims(x2[samplelist],axis=1)))
rt_train_noise = p/(1+e*np.cos(ytrain[:,1:2]))
rt_train_noisefree = p/(1+e*np.cos(ytrain_noisefree[:,1:2]))
print('Data variance:',np.var(rt_train_noisefree))
print('Perturbation variance:',np.var(rt_train_noisefree-rt_train_noise))
print(np.var(rt_train_noisefree-rt_train_noise)/np.var(rt_train_noisefree))
### Build a GP to infer the hyperparameters for each dynamic equation 
### and compute d_i conditioning on u_i
ytrain_hat = []
kernellist = []
GPlist = []

for i in range(0,NumDyn):

    xtkernel = GPy.kern.RBF(input_dim=1, variance=100, lengthscale=100)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)

    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)

    ypred,yvar = xtGP.predict(Xtrain)

    plt.plot(Xtrain,ytrain[:,i:(i+1)],'*',label='GT')
    plt.plot(Xtrain,ypred,'*',label='prediction')
    plt.legend()
    plt.savefig("GPdynamics"+str(i)+".png")
    plt.clf()

    ytrain_hat.append(ypred)
    kernellist.append(xtkernel)
    GPlist.append(xtGP)

ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T

invRdd_list = []
d_hat_list = []

### Truncate out the boundary data for better inference
# sort_Xtrain = np.sort(Xtrain,axis=0)
# sort_index = np.argsort(Xtrain,axis=None)
# sort_ytrain = ytrain[sort_index,:]
# sort_ytrain_hat = ytrain_hat[sort_index,:]

# Xtrain = sort_Xtrain[300:-300,:]
# ytrain = sort_ytrain[300:-300,:]
# ytrain_hat = sort_ytrain_hat[00:-100,:]


### Print out hyperparameters from a GP of x(t)
for i in range(0,NumDyn):
    GPvariance = kernellist[i][0]
    GPlengthscale = kernellist[i][1]
    GPnoise = GPlist[i]['Gaussian_noise'][0][0]
    print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

    ### Construct the covariance matrix 
    if NoisePer == 0:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*1e-7
        Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-7
    else:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    
        Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise

    Kdu = kernellist[i].dK_dX(Xtrain,Xtrain,0)
    Kud = Kdu.T
    invKuu = np.linalg.inv(Kuu)                                                  

    invRdd = Kdd-Kdu@invKuu@Kud
    Rdd = np.linalg.inv(invRdd)

    ### Compute the true value of d_i using GP
    d_hat = Kdu@invKuu@ytrain[:,i:i+1]

    ### Save d_hat and invRdd for likelihood construction
    invRdd_list.append(invRdd)
    d_hat_list.append(d_hat)

derivative_phi = (x1[1:]-x1[:-1])/1
derivative_chi = (x2[1:]-x2[:-1])/1

plt.plot(x1[1:],derivative_phi,'k-',label='GT')
plt.plot(x2[1:],derivative_chi,'k-')
plt.plot(ytrain_hat[:,0:1],d_hat_list[0],'.',label='GP - phi')
plt.plot(ytrain_hat[:,1:2],d_hat_list[1],'.',label='GP - chi')
# plt.xlim([0,50])
plt.legend()
plt.savefig('derivatve.png')
plt.clf()

##############################################
################# Build model ################
##############################################

### Construct model
with pm.Model() as model:

    ### prior of two parameters
    e_prior = pm.Normal('e', 10,sigma=10)
    p_prior = pm.Normal('p', 50,sigma=20)
    
    ### Model of phi
    phi_dot = (p_prior-2-2*e_prior*pm.math.cos(ytrain_hat[:,1:2]))*pm.math.sqr(1+e_prior*pm.math.cos(ytrain_hat[:,1:2]))\
                /(1*(p_prior**1.5))/pm.math.sqrt(pm.math.sqr(p_prior-2)-4*pm.math.sqr(e_prior))
    
    chi_dot = (p_prior-2-2*e_prior*pm.math.cos(ytrain_hat[:,1:2]))*pm.math.sqr(1+e_prior*pm.math.cos(ytrain_hat[:,1:2]))*pm.math.sqrt(p_prior-6-2*e_prior*pm.math.cos(ytrain_hat[:,1:2]))\
                /(1*(p_prior**2))/pm.math.sqrt(pm.math.sqr(p_prior-2)-4*pm.math.sqr(e_prior))
    

    Phi_obs = pm.MvNormal('phi_obs', mu=phi_dot, cov=invRdd_list[0], observed=d_hat_list[0]) 
    Chi_obs = pm.MvNormal('chi_obs', mu=chi_dot, cov=invRdd_list[1], observed=d_hat_list[1])
    
    step = pm.Metropolis()
    trace = pm.sample(PosteriorSample,step=step, return_inferencedata=False,cores=4,tune=1000,random_seed=0) ### tuning = 1000 (warm up period)
    # trace = pm.sample(PosteriorSample, return_inferencedata=False,cores=4,tune=PosteriorSample,random_seed=0)
    # approx = pm.fit(100000,method='advi',random_seed=0) ### VI


posterior_samples_e = np.squeeze(trace.get_values("e", combine=True))
posterior_samples_p = np.squeeze(trace.get_values("p", combine=True))

print('Mean (e): ', np.mean(posterior_samples_e))
print('STD (e): ', np.std(posterior_samples_e))
print('Mean (p): ',np.mean(posterior_samples_p))
print('STD (p): ',np.std(posterior_samples_p))

### Plot posterior distribution of parameters
plt.clf()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2, 3), sharex=False)
params = {
            'axes.labelsize': 18,
            'xtick.labelsize': 30,
            'ytick.labelsize': 18,
            'text.usetex': False,
            'axes.formatter.useoffset': True,
            'axes.formatter.use_mathtext': True
        }
plt.rcParams.update(params)

for term, ax in enumerate(axes):
    if term == 0:
        sns.kdeplot(posterior_samples_e, ax=ax,bw_adjust=3, color = 'grey',linewidths=2.5,fill=True)
        ax.axvline(0.5,linestyle='--',linewidth=2,color='black')
        ax.set_xlim(e-1e-1,e+1e-1)
        ax.set_ylabel('',fontsize=18)
        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.locator_params(axis='x', nbins=3)
        ax.tick_params(axis='x', which='major', labelsize=14)
        plt.tight_layout()
        #plt.savefig('distribution.png',bbox_inches='tight',transparent=True)
    else:
        sns.kdeplot(posterior_samples_p, ax=ax,bw_adjust=3, color = 'grey',linewidths=2.5,fill=True)
        ax.axvline(100,linestyle='--',linewidth=2,color='black')
        ax.set_xlim(p-1e1,p+1e1)
        ax.set_ylabel('',fontsize=18)
        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.locator_params(axis='x', nbins=3)
        ax.tick_params(axis='x', which='major', labelsize=14)
        plt.tight_layout()
plt.savefig('distribution_D'+str(NumDataTrain)+'_N'+str(int(NoisePer*1000))+'.png',bbox_inches='tight',transparent=True)



### Compute and plot time integration using inferred parameters
dt = 100
T = 6e4

### different IC:
###     1) phi_0 = np.pi/2, chi_0 = np.pi/2
###     2) 
phi_0 = np.pi/2
chi_0 = np.pi/2

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))
xlist = BBHmodel([phi_0,chi_0],T,dt,[p,e])

rt = p/(1+e*np.cos(xlist[1]))
xt = -rt*np.cos(xlist[0]) 
yt = -rt*np.sin(xlist[0])


xt_pred_list = []
yt_pred_list = []
for i in range(posterior_samples_e.shape[0]):
    if i%100==0:
        print('Working on sample:',i)
    phi_pred_temp, chi_pred_temp = BBHmodel([phi_0,chi_0],T,dt,[posterior_samples_p[i],posterior_samples_e[i]])
    rt_pred_temp = posterior_samples_p[i]/(1+posterior_samples_e[i]*np.cos(chi_pred_temp))
    xt_pred_temp = -rt_pred_temp*np.cos(phi_pred_temp) 
    yt_pred_temp = -rt_pred_temp*np.sin(phi_pred_temp)   
    
    xt_pred_list.append(xt_pred_temp)
    yt_pred_list.append(yt_pred_temp)

xt_pred_list = np.asarray(xt_pred_list)
yt_pred_list = np.asarray(yt_pred_list)

# rt_train = p/(1+e*np.cos(ytrain[:,1]))
# xt_train = -rt_train*np.cos(ytrain[:,0]) 
# yt_train = -rt_train*np.sin(ytrain[:,0])


plt.figure(figsize=(10, 3))
params = {
            'axes.labelsize': 17,
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
plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
plt.plot(time,xt,'k-',linewidth=2,label ='ground truth')
plt.plot(time,yt,'k-',linewidth=2)
plt.plot(time,np.mean(xt_pred_list,axis=0),'--',color='tab:blue',linewidth=2,label =r'$r_x$')
plt.plot(time,np.mean(yt_pred_list,axis=0),'--',color='darkorange',linewidth=2,label =r'$r_y$')
plt.fill_between(time,np.mean(xt_pred_list,axis=0)+np.std(xt_pred_list,axis=0),np.mean(xt_pred_list,axis=0)-np.std(xt_pred_list,axis=0),color='royalblue',alpha=0.5,label=r'$r_x$ uncertainty')
plt.fill_between(time,np.mean(yt_pred_list,axis=0)+np.std(yt_pred_list,axis=0),np.mean(yt_pred_list,axis=0)-np.std(yt_pred_list,axis=0),color='tab:orange',alpha=0.5,label=r'$r_y$ uncertainty')

plt.xlim([0,6e4])
plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
plt.savefig('Prediction_D'+str(NumDataTrain)+'_N'+str(int(NoisePer*1000))+'.png',bbox_inches='tight')
plt.clf()




ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')
for i in range(posterior_samples_e.shape[0]):
    ax.plot(xt_pred_list[i,:],yt_pred_list[i,:],time,'-',color='grey',alpha=0.2,linewidth=2,label =r'\boldsymbol{r}(t)')
ax.plot(xt, yt, time,linewidth=3, color='crimson',label='ground truth')
plt.locator_params(nbins=4)
plt.gca().ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
# ax.set_axis_off()
plt.savefig('3DPrediction_D'+str(NumDataTrain)+'_N'+str(int(NoisePer*1000))+'.png',transparent=True)
# plt.figure(figsize=(8, 8))
# plt.plot(xt_pred_list[-1,:],yt_pred_list[-1,:],'.',color='grey',alpha=1,linewidth=2,label =r'\boldsymbol{r}(t)')
# plt.plot(xt, yt,'.',color='crimson',label='ground truth')
# plt.plot([xt[-1],xt_pred_list[-1,-1]],[yt[-1],yt_pred_list[-1,-1]],'-',color='blue',label='ground truth')
# plt.savefig('2DPrediction_D'+str(NumDataTrain)+'_N'+str(NoisePer*100)+'.png',bbox_inches='tight')


# plt.clf()
# plt.figure(figsize=(5, 4))
# params = {
#             'axes.labelsize': 21,
#             'font.size': 21,
#             'legend.fontsize': 23,
#             'xtick.labelsize': 15,
#             'ytick.labelsize': 15,
#             'text.usetex': False,
#             'axes.linewidth': 2,
#             'xtick.major.width': 2,
#             'ytick.major.width': 2,
#             'xtick.major.size': 2,
#             'ytick.major.size': 2,
#         }
# plt.rcParams.update(params)

# if legend_switch == 0:
#     plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')
# plt.plot(timedata,prediction_mean,'--',color='tab:blue',linewidth=3,label=r'$x$ prediction')
# plt.fill_between(timedata,prediction_mean+prediction_std,prediction_mean-prediction_std,color='tab:blue',alpha=0.4,label=r'$x$ uncertainty')
# plt.scatter(Xtrain,ytrain,marker='X',s=80,color='tab:blue',edgecolors='k',label='training data',zorder=2)
# if legend_switch == 1:
#     plt.plot(timedata,x,'-k',linewidth=3,label='ground truth')
#     plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=2,frameon=False)

# # plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')
# plt.grid(alpha=0.5)
# # plt.ylim([0,1.2])
# plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(num_train))+'.png')

