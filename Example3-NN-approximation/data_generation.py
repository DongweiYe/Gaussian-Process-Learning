import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
alpha = -1
beta = 1
dt = 1e-3
T = 5

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))

x_t0 = 0.1 ### initial condition
xlist = ODEmodel(x_t0,T,dt,[alpha,beta])
np.save('data/ODEdata_0.npy',xlist)
np.save('data/time.npy',time)


x_t0 = 0.4 ### initial condition
xlist = ODEmodel(x_t0,T,dt,[alpha,beta])
np.save('data/ODEdata_1.npy',xlist)

x_t0 = 0.75 ### initial condition
xlist = ODEmodel(x_t0,T,dt,[alpha,beta])
np.save('data/ODEdata_2.npy',xlist)

plt.plot(time,xlist)
plt.savefig('data.png',bbox_inches='tight')