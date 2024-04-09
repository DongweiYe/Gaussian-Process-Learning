import numpy as np
import GPy
import pickle
import matplotlib.pyplot as plt
from pysindy.differentiation import SmoothedFiniteDifference,FiniteDifference
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from LotkaVolterra_model import *
np.random.seed(0)


######################################
############## Scenario 1 ############
######################################

### Parameters
TrainRatio = 0.4         ### Train/Test data split ratio
DataSparsity = 0.25      ### Take 25% of as the total data we have
NoiseMean = 0            ### 0 mean for white noise
NoisePer = 0.2             ### (0 to 1) percentage of noise. NoisePer*average of data = STD of white noise
NumDyn = 2               ### number of dynamics equation
PosteriorSample = 1000    ### posterior sampling numbers
IC_test = 0               ### test on different initial condition

time_integration = False   ### perform reconstruction based on inferred parameters
plotfunc = False           ### plot the reconstruction result
comparefunc = True        ### Enable comparsion (linear regression with finite difference)
smooth_tpye = 'SG'    ### None, SG or Spline,   
save_data = True

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
print('Data for training:',num_train)

### Define training data 
Xtrain = np.expand_dims(timedata[samplelist],axis=1)
ytrain = np.hstack((np.expand_dims(preydata[samplelist],axis=1),np.expand_dims(preddata[samplelist],axis=1)))
ytrain_backup = np.hstack((np.expand_dims(x1[samplelist],axis=1),np.expand_dims(x2[samplelist],axis=1)))


plt.plot(Xtrain,ytrain[:,0],'*',label='x1 dynamics')
plt.plot(Xtrain,ytrain[:,1],'*',label='x2 dynamics')
plt.plot(timedata,x1,'-k',label='x1')
plt.plot(timedata,x2,'-k',label='x2')
plt.legend()
plt.savefig("train_data_and_latent_dynamics.png")
plt.clf()

### Build a GP to infer the hyperparameters for each dynamic equation 
### and proper G_i data from regression
ytrain_hat = []
kernellist = []
GPlist = []

for i in range(0,NumDyn):
    xtkernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    xtGP = GPy.models.GPRegression(Xtrain,ytrain[:,i:(i+1)],xtkernel)

    xtGP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
    xtGP.optimize_restarts(num_restarts=2,verbose=False)
    
    ypred,yvar = xtGP.predict(Xtrain)

    plt.plot(Xtrain,ypred,'o',label='prediction')
    plt.plot(Xtrain,ytrain[:,i:(i+1)],'*',label='GT')
    plt.legend()
    plt.savefig("GPdynamics"+str(i)+".png")
    plt.clf()
    
    ytrain_hat.append(ypred)
    kernellist.append(xtkernel)
    GPlist.append(xtGP)
    
ytrain_hat = np.squeeze(np.asarray(ytrain_hat)).T

### loop for the estimation of each dynamic equation
para_mean = []
para_cova = []


for i in range(0,NumDyn):
    
    print('Estimate parameters in equation: ',i)

    ### Compute hyperparameters from a GP of x(t)
    GPvariance = kernellist[i][0]
    GPlengthscale = kernellist[i][1]
    GPnoise = GPlist[i]['Gaussian_noise'][0][0]
    print('GP hyperparameters:',GPvariance,GPlengthscale,GPnoise)    

    ### Construct the covariance matrix of equation,
    ### Due to the ill condition caused the periodic data when dealing with noise-free and large data scenarios
    ### Add manual noise on the diagonal to get small condition number
    if NoisePer == 0:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*1e-5
        Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*1e-5
    else:
        Kuu = kernellist[i].K(Xtrain) + np.identity(Xtrain.shape[0])*GPnoise                    
        Kdd = kernellist[i].dK2_dXdX2(Xtrain,Xtrain,0,0) + np.identity(Xtrain.shape[0])*GPnoise 

    Kdu = kernellist[i].dK_dX(Xtrain,Xtrain,0)                                                  
    Kud = Kdu.T                                                                                 
    invKuu = np.linalg.inv(Kuu)                                                  

    ### If we could only assume that Kuu is invertalbe, then
    Rdd = np.linalg.inv(Kdd-Kdu@invKuu@Kud)
    Rdu = -Rdd@Kdu@invKuu
    Rud = Rdu.T

    # print(np.linalg.cond(Kuu))
    # print(np.linalg.det(Kuu))
    # print(np.linalg.matrix_rank(Kuu))

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

if save_data == True:
    np.save('result/parameter/Mean_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_mean)))
    np.save('result/parameter/Cov_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_cova)))

### Prediction with marginalization
preylist_array = []
predlist_array = []

### Sample from parameter posterior and perform reconstruction and prediction via time integration
if time_integration == True:
    for i in range(PosteriorSample):

        mu1 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[0]),para_cova[0],1))
        mu2 = np.squeeze(np.random.multivariate_normal(np.squeeze(para_mean[1]),para_cova[1],1))

        ### LV other parameters
        if IC_test == 0:
            x1_t0 = 1
            x2_t0 = 1
            T = 20
        else:
            x1_t0 = 2
            x2_t0 = 1.2
            T = 10

        dt = 1e-3

        preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[mu1[0],-mu1[1],mu2[0],-mu2[1]])
        if IC_test == 0:
            if np.max(preylist) > 20 or np.max(predatorlist) > 20:
                pass
            else:
                preylist_array.append(preylist)
                predlist_array.append(predatorlist)
        else:
            preylist_array.append(preylist)
            predlist_array.append(predatorlist)


    #### Addition result for different initial conditions
    if IC_test == 1:
        preylist_IC,predatorlist_IC = LVmodel(x1_t0,x2_t0,T,dt,[1.5,1,1,3])


    preymean = np.mean(np.asarray(preylist_array),axis=0)
    predmean = np.mean(np.asarray(predlist_array),axis=0)
    preystd = np.std(np.asarray(preylist_array),axis=0)
    predstd = np.std(np.asarray(predlist_array),axis=0)


### This function is used to perform OpInf to compare the result
if comparefunc == True:
    def ridge_reg(g_data,d_data,regu):
        regressor = Ridge(alpha = regu)
        regressor.fit(g_data,d_data)
        # print('Regu parameter:',regressor.coef_)
        return regressor.coef_
    

    ### Rearrange the random data to time sequences
    sort_Xtrain = np.sort(Xtrain,axis=0)
    sort_index = np.argsort(Xtrain,axis=None)
    sort_ytrain = ytrain[sort_index,:]
    para_OpInf = []
    

    ### denoise by total variation regularization (smoothing)
    if NoisePer != 0:
        ### False = no-smoothing
        ### SFD = Smooth via total variation and differentiation
        ### Kalman = kalman filter
        if smooth_tpye == 'None':
            delta_y = sort_ytrain[1:,:]-sort_ytrain[:-1,:]
            delta_t = sort_Xtrain[1:,:]-sort_Xtrain[:-1,:]
            d_hat = np.divide(delta_y,delta_t)
            y_smooth = sort_ytrain[1:,:]

        elif smooth_tpye == 'SG':
            # y_smooth = savgol_filter(sort_ytrain, window_length=51,polyorder=3, axis=0)
            # # delta_y = y_smooth[1:,:]-y_smooth[:-1,:]
            # # delta_t = sort_Xtrain[1:,:]-sort_Xtrain[:-1,:]
            # # d_hat = np.divide(delta_y,delta_t)
            # # y_smooth = y_smooth[1:,:]
            neigh_inte = 0.3
            d_hat = ps.SINDyDerivative(kind="savitzky_golay", left=neigh_inte, right=neigh_inte, order=3)._differentiate(sort_ytrain,sort_Xtrain[:,0])
            sfd = SmoothedFiniteDifference(axis=0)
            y_smooth = sfd.smoother(sort_ytrain,window_length=11,polyorder=3,axis=0)  ### need cases to cases tuning 
            # d_hat = sfd._differentiate(sort_ytrain,sort_Xtrain[:,0])/2

        # elif smooth_tpye == 'TV':
        #     d_hat = ps.SpectralDerivative()._differentiate(sort_ytrain,sort_Xtrain[:,0])
        #     y_smooth = sort_ytrain

        
        plt.clf()
        plt.plot(sort_Xtrain,sort_ytrain,label='data')
        plt.plot(sort_Xtrain,y_smooth,label='smoothed data')
        plt.plot(timedata[:8000],x1[:8000],'k-')
        plt.plot(timedata[:8000],x2[:8000],'k-') 
        plt.legend()       
        plt.savefig('smoothed_data.png')

        # plt.clf()
        # plt.plot(sort_Xtrain,d_hat,'X',label='data')
        # # plt.plot(sort_Xtrain,y_smooth,'*',label='smoothed data')
        # plt.plot(timedata[:8000],1.5*x1[:8000] - x1[:8000]*x2[:8000],'k-')
        # plt.plot(timedata[:8000],x2[:8000]*x1[:8000] - 3*x2[:8000],'k-') 
        # plt.legend()       
        # plt.savefig('smoothed_dhat.png')

        

    else:
        ### Finite different to comput the derivative for noise-free data (without smoothing)
        delta_y = sort_ytrain[1:,:]-sort_ytrain[:-1,:]
        delta_t = sort_Xtrain[1:,:]-sort_Xtrain[:-1,:]
        d_hat = np.divide(delta_y,delta_t)
        y_smooth = sort_ytrain[1:,:]

        plt.clf()
        plt.plot(sort_Xtrain,sort_ytrain,'*',label='data')
        plt.plot(sort_Xtrain[1:,:],y_smooth,'*',label='smoothed data')
        plt.plot(timedata[:8000],x1[:8000],'k-')
        plt.plot(timedata[:8000],x2[:8000],'k-') 
        plt.legend()       
        plt.savefig('smoothed_data.png')


    ### Inference
    for i in range(0,NumDyn):
        if i == 0:
            G = np.hstack((y_smooth[:,0:1],np.multiply(y_smooth[:,0:1],y_smooth[:,1:2])))
        else:
            G = np.hstack((np.multiply(y_smooth[:,0:1],y_smooth[:,1:2]),y_smooth[:,1:2]))

        theta_compare = ridge_reg(G,d_hat[:,i:i+1],1)        
        para_OpInf.append(theta_compare)
        print('OpInf prediciton:',theta_compare)
    

    if save_data == True:
        np.save('result/parameter/'+str(smooth_tpye)+'_N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.npy',np.squeeze(np.asarray(para_OpInf)))



### This function is used to plot the reconstruction/prediction of the dynamics over time
if plotfunc == True:
    if IC_test == 1:
        plt.figure(figsize=(9, 2))
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
        new_timedata = np.arange(0,T+T/(T/dt)*0.1,T/(T/dt))
        plt.plot(new_timedata,preylist_IC,'-k',linewidth=3,label='ground truth')
        plt.plot(new_timedata,predatorlist_IC,'-k',linewidth=3)

        plt.plot(new_timedata,preymean,'--',color='royalblue',linewidth=3,label=r'$x_1$ prediction')
        plt.plot(new_timedata,predmean,'--',color='tab:orange',linewidth=3,label=r'$x_2$ prediction')

        plt.fill_between(new_timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5,label=r'$x_1$ uncertainty')
        plt.fill_between(new_timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5,label=r'$x_2$ uncertainty')
        plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
        # plt.show()
        plt.savefig('result/figure/ICs_'+str(x1_t0)+'&'+str(x2_t0)+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')

    else:
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

        plt.plot(timedata,x1,'-k',linewidth=3,label='ground truth')
        plt.plot(timedata,x2,'-k',linewidth=3)

        plt.plot(timedata,preymean,'--',color='royalblue',linewidth=3,label=r'$x_1$ prediction')
        plt.plot(timedata,predmean,'--',color='tab:orange',linewidth=3,label=r'$x_2$ prediction')

        plt.fill_between(timedata,preymean+preystd,preymean-preystd,color='royalblue',alpha=0.5,label=r'$x_1$ uncertainty')
        plt.fill_between(timedata,predmean+predstd,predmean-predstd,color='tab:orange',alpha=0.5,label=r'$x_2$ uncertainty')

        plt.scatter(Xtrain,ytrain[:,0],marker='X',s=80,color='royalblue',edgecolors='k',label='training data '+r'($x_1$)',zorder=2.5)
        plt.scatter(Xtrain,ytrain[:,1],marker='X',s=80,color='darkorange',edgecolors='k',label='training data '+r'($x_2$)',zorder=2.5)

        plt.axvline(timedata[-1]*TrainRatio,linestyle='-',linewidth=3,color='grey')

        if NoisePer == 0:
            plt.ylim([-0.8,8])
        # plt.xlim([-1,20])
        # plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=4,frameon=False)
        # plt.show()
        plt.savefig('result/figure/N'+str(int(NoisePer*100))+'D'+str(int(DataSparsity*400))+'.png',bbox_inches='tight')
    