import numpy as np
import matplotlib.pyplot as plt

NoisePer = [0,10,20]
DataSparsity = [100,10,5,1]
gt = np.array([1.5,-1,1,-3])

gandy_list = []
none_list = []
sg_list = []
kalman_list = []

for i in range(len(NoisePer)):
    gandy_list_noise = []
    none_list_noise = []
    sg_list_noise = []
    kalman_list_noise = []

    for j in range(len(DataSparsity)):
        gandy_list_noise.append(np.load('result/parameter/Mean_N'+str(NoisePer[i])+'D'+str(DataSparsity[j])+'.npy'))
        sg_list_noise.append(np.load('result/parameter/SG_N'+str(NoisePer[i])+'D'+str(DataSparsity[j])+'.npy'))


    gandy_list.append(gandy_list_noise)
    sg_list.append(sg_list_noise)

###  gandy_list[noise][data][[alpha, -beta],[delta,-gamma]]
noise_index = 2
element_index = 3
legend = 0

gandy_result = []
none_result = []
sg_result = []
kalman_result = []
for i in range(len(DataSparsity)):
    gandy_result.append(gandy_list[noise_index][i].flatten()[element_index])
    sg_result.append(sg_list[noise_index][i].flatten()[element_index])
    # kalman_result.append(kalman_list[noise_index][i].flatten()[element_index])
    # none_result.append(none_list[noise_index][i].flatten()[element_index])




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
plt.semilogy(x,np.abs(gandy_result-gt[element_index])/np.abs(gt[element_index])*100,'-X',color='tab:blue',linewidth=4,markersize=15,label='GPL')
plt.semilogy(x,np.abs(sg_result-gt[element_index])/np.abs(gt[element_index])*100,'-X',color='tab:orange',linewidth=4,markersize=15,label='FD+LinReg')
# if noise_index != 0:
#     plt.semilogy(x,np.abs(sg_result-gt[element_index])/np.abs(gt[element_index])*100,'-X',color='tab:red',linewidth=4,markersize=15,label='FD+LinReg (Savitzky-Golay)')
#     plt.semilogy(x,np.abs(kalman_result-gt[element_index])/np.abs(gt[element_index])*100,'-X',color='tab:purple',linewidth=4,markersize=15,label='FD+LinReg (Kalman)')
    
if legend == 1:
    plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.,frameon=False)
if noise_index == 2:
    plt.xlabel('data density (%)')
if element_index == 0:
    plt.ylabel('relative error (%)')
label = [100, 10, 5, 1]
plt.ylim([8e-3,3e3])
plt.xticks(x, label)
plt.grid()
plt.savefig('compare_N'+str(noise_index)+'D'+str(element_index)+'.png',bbox_inches='tight')