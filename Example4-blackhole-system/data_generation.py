import numpy as np
import matplotlib.pyplot as plt
from dynamical_system import *



################## The code is used to generate the data for 1D dynamical system  
################## for senario 3 (neurl ODE)

### Definite parameters
e = 0.5
p = 100

phi_0 = 0
chi_0 = np.pi

dt = 1
T = 1e4

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))
xlist = BBHmodel([phi_0,chi_0],T,dt,[p,e])

rt = p/(1+e*np.cos(xlist[1]))
xt = -rt*np.cos(xlist[0]) 
yt = -rt*np.sin(xlist[0])

np.save('data/BBH_x1.npy',xlist[0])
np.save('data/BBH_x2.npy',xlist[1])
np.save('data/time.npy',time)



# plt.plot(time,xlist[0])
# plt.plot(time,xlist[1])
# plt.savefig('data.png',bbox_inches='tight')

# plt.clf()
# plt.plot(xt,yt)
# plt.xlim([-200,200])
# plt.ylim([-200,200])
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.savefig('trajectory.png',bbox_inches='tight')




