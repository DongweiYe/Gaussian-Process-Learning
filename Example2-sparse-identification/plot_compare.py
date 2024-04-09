import numpy as np
import matplotlib.pyplot as plt

NoisePer = [0,10,20]
DataSparsity = [100,10,5,1]
element_list = [1,5,8,11]
gt = np.array([0,1.5,0,0,0,-1,0,0,-3,0,0,1])

gandy_list = []
sindy_list = []

for i in range(len(NoisePer)):
    gandy_list_temp = []
    sindy_list_temp = []

    for j in range(len(DataSparsity)):
        gandy_list_temp.append(np.load('result/parameter/Mean_D'+str(DataSparsity[j])+'_N'+str(NoisePer[i])+'.npy'))
        sindy_list_temp.append(np.load('result/parameter/SINDy_N'+str(NoisePer[i])+'D'+str(DataSparsity[j])+'.npy'))

    gandy_list.append(gandy_list_temp)
    sindy_list.append(sindy_list_temp)


###  gandy_list[noise][data][[alpha, -beta],[delta,-gamma]]
noise_index = 0
element_index = 0   
legend = 1

gandy_result = []
sindy_result = []


for i in range(len(DataSparsity)):
    gandy_result.append(gandy_list[noise_index][i].flatten()[element_list[element_index]])
    sindy_result.append(sindy_list[noise_index][i].flatten()[element_list[element_index]])


plt.figure(figsize=(5, 5))
params = {
        'axes.labelsize': 21,
        'font.size': 21,
        'legend.fontsize': 23,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'text.usetex': False,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
    }
plt.rcParams.update(params)
x = [0,1,2,3]
plt.semilogy(x,np.abs(gandy_result-gt[element_list[element_index]])/np.abs(gt[element_list[element_index]])*100,'-X',color='tab:blue',linewidth=4,markersize=15,label='GPL')
if noise_index == 0:
    plt.semilogy(x,np.abs(sindy_result-gt[element_list[element_index]])/np.abs(gt[element_list[element_index]])*100,'-X',color='tab:orange',linewidth=4,markersize=15,label='SINDy')
else:
    plt.semilogy(x,np.abs(sindy_result-gt[element_list[element_index]])/np.abs(gt[element_list[element_index]])*100,'-X',color='tab:orange',linewidth=4,markersize=15,label='ensemble SINDy')

    
if legend == 1:
    plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.,frameon=False)
if noise_index == 2:
    plt.xlabel('data density (%)')
if element_index == 0:
    plt.ylabel('relative error (%)')

label = [100, 10, 5, 1]

plt.ylim([2e-2,8e2])
plt.xticks(x, label)
plt.grid()
plt.savefig('compare_N'+str(noise_index)+'D'+str(element_index)+'.png',bbox_inches='tight')