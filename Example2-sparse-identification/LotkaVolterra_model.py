import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def LVmodel(x1_initial,x2_initial,total_time,dt,modelparameter):
    
    preylist = [x1_initial]
    predatorlist = [x2_initial]

    num_T = int(total_time/dt)

    pre_prey = x1_initial
    pre_pred = x2_initial

    def model_derivative(t, state, a, b, c, d):
        x,y = state
        return [a*x-b*x*y, c*x*y-d*y]

    sol = solve_ivp(model_derivative, [0, total_time], [pre_prey, pre_pred],\
                    args=(modelparameter[0], modelparameter[1], modelparameter[2], modelparameter[3]),\
                    method='BDF',t_eval=np.linspace(0,total_time,num_T+1))
    
    return sol.y[0,:], sol.y[1,:]
    # for timestep in range(num_T):
    #     next_prey = dt*pre_prey*(modelparameter[0] - modelparameter[1] * pre_pred) + pre_prey
    #     next_pred = dt*pre_pred*(modelparameter[2] * pre_prey - modelparameter[3]) + pre_pred

    #     preylist.append(next_prey)
    #     predatorlist.append(next_pred)
        
    #     pre_prey = next_prey
    #     pre_pred = next_pred

    # return np.array(preylist),np.array(predatorlist)

