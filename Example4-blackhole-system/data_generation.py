import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
e = 0.5
p = 100
M = 1
phi_0 = 0
chi_0 = np.pi

dt = 1
T = 6e4

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))
xlist = ODEmodel([phi_0,chi_0],T,dt,[e,p,M])

plt.plot(time,xlist)
plt.savefig('data.png',bbox_inches='tight')

# np.save('data/ODEdata_0.npy',xlist)
# np.save('data/time.npy',time)

