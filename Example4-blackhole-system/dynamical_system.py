import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def BBHmodel(x_initial,total_time,dt,modelparameter):
    
    num_T = int(total_time/dt)

    def model_derivative(t, state, p,e):
        phi,chi = state

        chi_dynamics = (p-2-2*e*np.cos(chi))*np.square(1+e*np.cos(chi))*np.sqrt(p-6-2*e*np.cos(chi))/(np.square(p))/np.sqrt(np.square(p-2)-4*np.square(e))
        phi_dynamics = (p-2-2*e*np.cos(chi))*np.square(1+e*np.cos(chi))/(np.power(p,1.5))/np.sqrt(np.square(p-2)-4*np.square(e))
        
        return [phi_dynamics, chi_dynamics]

    sol = solve_ivp(model_derivative, [0, total_time], x_initial,\
                    args=(modelparameter[0], modelparameter[1]),\
                    method='BDF',t_eval=np.linspace(0,total_time,num_T+1))
    
    return sol.y[0,:], sol.y[1,:]