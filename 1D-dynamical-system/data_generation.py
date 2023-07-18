import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
x_t0 = 1 ### initial condition
x1_t = -0.1

alpha = 2
beta = -1
gamma = -0.3

dt = 1e-3
T = 5

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))
### Example 1
# xlist = tanh_model(x_t0,T,dt,[alpha,beta,gamma])
# x1list = tanh_model(x1_t,T,dt,[alpha,beta,gamma])

### Example 2 & 3
xlist = sinexp_model(x_t0,T,dt)
# xlist = xsinx_model(x_t0,T,dt)
# xlist = new_model(x_t0,T,dt)


plt.plot(time,xlist)
# plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),x1list)
plt.show()


np.save('data/e2_x_ic1.npy',xlist)
# np.save('data/xsinx_x_ic2.npy',x1list)
np.save('data/time.npy',time)
