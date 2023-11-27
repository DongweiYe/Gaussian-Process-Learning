import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LotkaVolterra_model import *


def pred_validation(regu_para,x1,x2,data_input,data_output,inferred_para):
    if len(inferred_para) != 2:
        return np.nan
    else:
        x1_t0 = 1 ### Prey initial
        x2_t0 = 1 ### Predator intial

        dt = 1e-3
        T = 8

        preylist,predatorlist = learned_model(x1_t0,x2_t0,T,dt,inferred_para)
        time = np.arange(0,T+T/(T/dt),T/(T/dt))

        search_array = np.searchsorted(time, data_input)
        prediction_1 = np.squeeze(preylist[search_array])
        prediction_2 = np.squeeze(predatorlist[search_array])

        error_integral = np.sum(np.abs(data_output[:,0]-prediction_1)) \
                        + np.sum(np.abs(data_output[:,1]-prediction_2))
        
        plt.title(str(inferred_para[0])+'\n'+str(inferred_para[1]))
        plt.plot(time,x1[:8001],'-',color='k',label='GT')
        plt.plot(time,x2[:8001],'-',color='k')
        # plt.plot(data_input,data_output[:,0],'o',label='data_1')
        # plt.plot(data_input,data_output[:,1],'o',label='data_2')
        plt.plot(data_input,prediction_1,'*',markersize=10,label='prediction_1')
        plt.plot(data_input,prediction_2,'*',markersize=10,label='prediction_2')
        plt.legend()
        plt.xlabel('error:'+str(error_integral))
        plt.savefig(str(regu_para)+'.png',bbox_inches='tight')
        plt.clf()

        return error_integral

def learned_model(x1_initial,x2_initial,total_time,dt,inferred_para):
    
    preylist = [x1_initial]
    predatorlist = [x2_initial]

    num_T = int(total_time/dt)

    pre_prey = x1_initial
    pre_pred = x2_initial

    a = inferred_para[0]
    b = inferred_para[1]
    # a[np.abs(a)<1e-1] = 0
    # b[np.abs(b)<1e-1] = 0
    # print(a)
    # print(b)
    for timestep in range(num_T):
        next_prey = dt*(a[0] + pre_prey*a[1] + pre_pred*a[2] \
                                 + (pre_prey**2)*a[3] + (pre_pred**2)*a[4] + (pre_prey*pre_pred)*a[5]) + pre_prey
        next_pred = dt*(b[0] + pre_prey*b[1] + pre_pred*b[2] \
                                 + (pre_prey**2)*b[3] + (pre_pred**2)*b[4] + (pre_prey*pre_pred)*b[5]) + pre_pred
    
        # next_prey = dt*(pre_prey*1.5 + (pre_prey*pre_pred)*(-1)) + pre_prey
        # next_pred = dt*(pre_pred*(-3)+ (pre_prey*pre_pred)*1) + pre_pred

        preylist.append(next_prey)
        predatorlist.append(next_pred)
        
        pre_prey = next_prey
        pre_pred = next_pred

    return np.array(preylist),np.array(predatorlist)