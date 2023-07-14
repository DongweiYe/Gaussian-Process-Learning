import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
x_t0 = 1.5 ### initial condition
x1_t = -0.1

alpha = 2
beta = -1
gamma = -0.3

dt = 1e-3
T = 6

xlist = tanh_model(x_t0,T,dt,[alpha,beta,gamma])
x1list = tanh_model(x1_t,T,dt,[alpha,beta,gamma])

plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),xlist)
plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),x1list)
# plt.show()

print(xlist.shape)

np.save('data/tanh_x_ic1.npy',xlist)
np.save('data/tanh_x_ic2.npy',x1list)
np.save('data/time.npy',np.arange(0,T+T/(T/dt),T/(T/dt)))