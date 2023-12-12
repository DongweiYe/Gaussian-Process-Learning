import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
x_t0 = 0.1 ### initial condition

alpha = -1
beta = 1


dt = 1e-3
T = 5

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))

### Example 
xlist = ODEmodel(x_t0,T,dt,[alpha,beta])


plt.plot(time,xlist)
plt.savefig('data.png',bbox_inches='tight')


np.save('data/ODEdata.npy',xlist)
np.save('data/time.npy',time)
