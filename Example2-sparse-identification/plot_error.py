import numpy as np
import matplotlib.pyplot as plt

DataSparsity = [0.025,0.05,0.075,0.125]
NoisePer = [0,0.05,0.1,0.15,0.2]
GT = np.array([0,1.5,0,0,0,-1,0,0,-3,0,0,1])
index = [1,5,8,11]
#####################################################################################################################

data_list_1_mean = []
data_list_1_vari = []

for i in range(5):
    mean = np.load('result/parameter/Mean_D'+str(int(DataSparsity[0]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = np.load('result/parameter/Vari_D'+str(int(DataSparsity[0]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()

    vari = vari[index]

    data_list_1_mean.append(mean)
    data_list_1_vari.append(vari)

print(data_list_1_mean)

error_list_1 = []
for i in range(5):
    error_list_1.append(np.sqrt((np.sum(np.square(GT-data_list_1_mean[i])))/np.sum(np.square(GT)))*100)

vari_list_1 = []
for i in range(5):
    vari_list_1.append(np.sqrt((np.sum(data_list_1_vari[i]))/np.sum(np.square(GT)))*100)



##########################################################################################################################
data_list_2_mean = []
data_list_2_vari = []

for i in range(5):
    mean = np.load('result/parameter/Mean_D'+str(int(DataSparsity[1]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = np.load('result/parameter/Vari_D'+str(int(DataSparsity[1]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = vari[index]
    data_list_2_mean.append(mean)
    data_list_2_vari.append(vari)

error_list_2 = []
for i in range(5):
    error_list_2.append(np.sqrt((np.sum(np.square(GT-data_list_2_mean[i])))/np.sum(np.square(GT)))*100)

vari_list_2 = []
for i in range(5):
    vari_list_2.append(np.sqrt((np.sum(data_list_2_vari[i]))/np.sum(np.square(GT)))*100)


#########################################################################################
data_list_3_mean = []
data_list_3_vari = []

for i in range(5):
    mean = np.load('result/parameter/Mean_D'+str(int(DataSparsity[2]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = np.load('result/parameter/Vari_D'+str(int(DataSparsity[2]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = vari[index]
    data_list_3_mean.append(mean)
    data_list_3_vari.append(vari)

error_list_3 = []
for i in range(5):
    error_list_3.append(np.sqrt((np.sum(np.square(GT-data_list_3_mean[i])))/np.sum(np.square(GT)))*100)

vari_list_3 = []
for i in range(5):
    vari_list_3.append(np.sqrt((np.sum(data_list_3_vari[i]))/np.sum(np.square(GT)))*100)

########################################################################################

data_list_4_mean = []
data_list_4_vari = []

for i in range(5):
    mean = np.load('result/parameter/Mean_D'+str(int(DataSparsity[3]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = np.load('result/parameter/Vari_D'+str(int(DataSparsity[3]*400))+'_N'+str(int(NoisePer[i]*100))+'.npy').flatten()
    vari = vari[index]
    data_list_4_mean.append(mean)
    data_list_4_vari.append(vari)

error_list_4 = []
for i in range(5):
    error_list_4.append(np.sqrt((np.sum(np.square(GT-data_list_4_mean[i])))/np.sum(np.square(GT)))*100)

vari_list_4 = []
for i in range(5):
    vari_list_4.append(np.sqrt((np.sum(data_list_4_vari[i]))/np.sum(np.square(GT)))*100)



# print(NoisePer)
plt.figure(figsize=(7, 5))
plt.rcParams["mathtext.fontset"] = 'cm'
params = {
        'axes.labelsize': 21,
        'font.size': 21,
        'legend.fontsize': 16,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'text.usetex': False,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.major.size': 1,
        'ytick.major.size': 1,
    }
plt.rcParams.update(params)
plt.plot(np.array(NoisePer)*100,np.array(error_list_1),'o-',linewidth=2.5,markersize=10,color='tab:orange', label='10% data')
plt.plot(np.array(NoisePer)*100,np.array(error_list_2),'o-',linewidth=2.5,markersize=10,color='tab:blue', label='20% data')
plt.plot(np.array(NoisePer)*100,np.array(error_list_3),'o-',linewidth=2.5,markersize=10,color='tab:red', label='30% data')
plt.plot(np.array(NoisePer)*100,np.array(error_list_4),'o-',linewidth=2.5,markersize=10,color='tab:purple', label='50% data')
plt.xlabel('noise level (%)')
plt.ylim([-0.3,11])
plt.ylabel(r'$\epsilon_1$ (%)')
plt.grid(alpha=0.5)
plt.legend()
plt.locator_params(axis='y', nbins=6) 
plt.gca().invert_xaxis()
plt.savefig('figure1.png',bbox_inches='tight')
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(np.array(NoisePer)*100,np.array(vari_list_1),'o-',linewidth=2.5,markersize=10,color='tab:orange', label='10% data')
plt.plot(np.array(NoisePer)*100,np.array(vari_list_2),'o-',linewidth=2.5,markersize=10,color='tab:blue', label='20% data')
plt.plot(np.array(NoisePer)*100,np.array(vari_list_3),'o-',linewidth=2.5,markersize=10,color='tab:red', label='30% data')
plt.plot(np.array(NoisePer)*100,np.array(vari_list_4),'o-',linewidth=2.5,markersize=10,color='tab:purple', label='50% data')
plt.xlabel('noise level (%)')
plt.ylabel(r'$\epsilon_2$ (%)')
plt.ylim([-0.3,11])
plt.grid(alpha=0.5)
plt.gca().invert_xaxis()
plt.legend()
plt.locator_params(axis='y', nbins=6)
plt.savefig('figure2.png',bbox_inches='tight')
plt.close()



print(error_list_1)
print(error_list_2)
print(error_list_3)

# print(data_list_3_mean[-1])
# print(vari_list_2)



# plt.plot(NoisePer,)
# print(data_list_1_mean)