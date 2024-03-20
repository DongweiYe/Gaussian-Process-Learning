import numpy as np
import matplotlib.pyplot as plt
from LotkaVolterra_model import *



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

plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),preylist)
plt.plot(np.arange(0,T+T/(T/dt),T/(T/dt)),predatorlist)
# plt.show()
plt.savefig('data1.png')

np.save('data/x1.npy',preylist)
np.save('data/x2.npy',predatorlist)
np.save('data/time.npy',np.arange(0,T+T/(T/dt),T/(T/dt)))