import numpy as np
import matplotlib.pyplot as plt



def LVmodel(x1_initial,x2_initial,total_time,dt,modelparameter):
    
    preylist = [x1_initial]
    predatorlist = [x2_initial]

    num_T = int(total_time/dt)

    pre_prey = x1_initial
    pre_pred = x2_initial
    for timestep in range(num_T):
        next_prey = dt*pre_prey*(modelparameter[0] - modelparameter[1] * pre_pred) + pre_prey
        next_pred = dt*pre_pred*(modelparameter[2] * pre_prey - modelparameter[3]) + pre_pred

        preylist.append(next_prey)
        predatorlist.append(next_pred)
        
        pre_prey = next_prey
        pre_pred = next_pred

    return np.array(preylist),np.array(predatorlist)



################## The code is used to generate the data for LV model  
################## for senario 1 (Operator inference) and 2 (SiNDy)

### Definite parameters
x1_t0 = 1 ### Prey initial
x2_t0 = 1 ### Predator intial

alpha = 1.5
beta = 1
delta = 1
gamma = 3

dt = 1e-3
T = 20

preylist,predatorlist = LVmodel(x1_t0,x2_t0,T,dt,[alpha,beta,delta,gamma])

# plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),preylist)
# plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),predatorlist)
# plt.show()

print(preylist.shape)

np.save('data/x1.npy',preylist)
np.save('data/x2.npy',predatorlist)
np.save('data/time.npy',np.arange(0,T+T/(T/dt),T/(T/dt)))
# print(preylist)