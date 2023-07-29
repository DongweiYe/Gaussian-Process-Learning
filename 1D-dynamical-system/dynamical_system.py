import numpy as np
import matplotlib.pyplot as plt



def tanh_model(x_initial,total_time,dt,modelparameter):
    
    xlist = [x_initial]
    num_T = int(total_time/dt)
    pre_x = x_initial
    for timestep in range(num_T):
        
        next_x = dt*modelparameter[2]*np.tanh(modelparameter[0]*pre_x+modelparameter[1]) + pre_x

        xlist.append(next_x)
        pre_x = next_x

    return np.array(xlist)

def sinexp_model(x_initial,total_time,dt):
    
    xlist = [x_initial]
    num_T = int(total_time/dt)
    pre_x = x_initial
    for timestep in range(num_T):
        
        next_x = dt*(2*np.sin(3*pre_x-1)-np.exp(pre_x/5+1)) + pre_x

        xlist.append(next_x)
        pre_x = next_x

    return np.array(xlist)

def sin_model(x_initial,total_time,dt,modelparameter):
    
    xlist = [x_initial]
    num_T = int(total_time/dt)
    pre_x = x_initial
    for timestep in range(num_T):
        
        next_x = dt*(modelparameter[2]*np.sin(modelparameter[0]*pre_x+modelparameter[1])) + pre_x

        xlist.append(next_x)
        pre_x = next_x

    return np.array(xlist)