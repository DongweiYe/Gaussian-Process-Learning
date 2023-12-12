import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import copy
def oneplot_dist(samples,name):
    
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(3, 8), sharex=True)

    # params = {
    #         'axes.labelsize': 20,
    #         'xtick.labelsize': 22,
    #         'ytick.labelsize': 22,
    #         'text.usetex': False,
    #     }
    # plt.rcParams.update(params)
    
    y_tick_labels = [r'$1$',r'$x_1$', r'$x_2$', r'$x_1^2$', r'$x_2^2$', r'$x_1 x_2$']

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


def multiplot_dist(samples,eqn,name):
    ### Get rid of negative sign before the parameters and offer GT for comparison
    plt.close()
    matplotlib.rcdefaults()
    plt.rcParams["mathtext.fontset"] = 'cm'
    if eqn == '0':
        gt_para = [0,1.5,0,0,0,1]
        samples[:,-1] = -samples[:,-1] 
    else:
        gt_para = [0,0,3,0,0,1]
        samples[:,2] = -samples[:,2]
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(3, 8), sharex=False)

    params = {
            'axes.labelsize': 18,
            'xtick.labelsize': 30,
            'ytick.labelsize': 18,
            'text.usetex': False,
            'axes.formatter.useoffset': True,
            'axes.formatter.use_mathtext': True
        }
    plt.rcParams.update(params)
    
    y_tick_labels = [r'$1$',r'$x_1$', r'$x_2$', r'$x_1^2$', r'$x_2^2$', r'$x_1 x_2$']

    for term, ax in enumerate(axes):
        
        if eqn == '0':
            sns.kdeplot(samples[:,term], ax=ax,bw_adjust=3, color = 'tab:blue',linewidths=2.5,fill=True)
        else:
            sns.kdeplot(samples[:,term], ax=ax,bw_adjust=3, color = 'tab:orange',linewidths=2.5 ,fill=True)
        ax.axvline(gt_para[term],linestyle='--',linewidth=2,color='black')
        if eqn == '0':
            ax.set_ylabel(y_tick_labels[term],fontsize=20)
        else:
            ax.set_ylabel(y_tick_labels[term],fontsize=20)
        ax.set_xlim(gt_para[term]-5e-1,gt_para[term]+5e-1)
        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        # formatter = mticker.ScalarFormatter(useMathText=True)
        # ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        ax.locator_params(axis='x', nbins=3)
        ax.tick_params(axis='x', which='major', labelsize=14)
        # ax.ticklabel_format(axis='x',style='scientific',useOffset=True,useMathText=True)
        # ax.xticks(fontsize=13)
    if eqn == '0':
        plt.xlabel(r'$\mathbf{\theta}_1$',fontsize=18,labelpad=20)
    else:
        plt.xlabel(r'$\mathbf{\theta}_2$',fontsize=18,labelpad=20)
    plt.tight_layout()
    
    # plt.legend(loc='upper left',bbox_to_anchor=(0.0, -0.5),ncol=3,frameon=False)
    # plt.show()
    plt.savefig('sparsity_inference_'+name+'.png',bbox_inches='tight',transparent=True)
    