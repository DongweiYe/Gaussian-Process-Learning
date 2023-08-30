import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sindy_dist(samples,name):
    
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(3, 8), sharex=True)

    # params = {
    #         'axes.labelsize': 20,
    #         'xtick.labelsize': 22,
    #         'ytick.labelsize': 22,
    #         'text.usetex': False,
    #     }
    # plt.rcParams.update(params)
    
    y_tick_labels = [r'$1$', r'$x_1$', r'$x_2$', r'$x_1^2$', r'$x_2^2$', r'$x_1 x_2$']

    for term, ax in enumerate(axes):
        sns.kdeplot(samples[:,term], ax=ax,bw_adjust=3, color = 'tab:blue',linewidths=1.5 ,fill=True)
        ax.set_ylabel(y_tick_labels[term],fontsize=15)
        ax.set_xlim(-3,3)
        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if term != 0:
            ax.spines['top'].set_visible(False)
        else:
            ax.spines['top'].set_linewidth(1.5)
        if term != 5:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left = False, bottom = False)
        else:
            ax.spines['bottom'].set_linewidth(1.5)
    # plt.axvline(0,linestyle='-',linewidth=3,color='black')
    plt.xticks(fontsize=13)

    plt.xlabel('posterior sample values',fontsize=15)
    plt.tight_layout()
    
    # plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
    # plt.show()
    plt.savefig('sparsity_inference_'+name+'.png',bbox_inches='tight')
