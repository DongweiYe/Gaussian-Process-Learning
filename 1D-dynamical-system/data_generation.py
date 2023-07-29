import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
x_t0 = 0 ### initial condition
x1_t = -0.1

alpha = 0.8
beta = 0.5
gamma = 2 

dt = 1e-3
T = 5

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))

### Example 0
# xlist = tanh_model(x_t0,T,dt,[alpha,beta,gamma])

### Example 1
xlist = sin_model(x_t0,T,dt,[alpha,beta,gamma])

plt.plot(time,xlist)
plt.show()


# np.save('data/tanh_x_ic.npy',xlist)
np.save('data/sin_x_ic.npy',xlist)

np.save('data/time.npy',time)
