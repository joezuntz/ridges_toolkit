import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import json

rc('text', usetex=False)
rc('font',**{'family':'serif','serif':['Times']})

def plot_prediction_frac_error(xarr, 
                               predicted_signal, 
                               true_signal, 
                               fractional_err, 
                               idx1=0, idx2=None, 
                               legend=None, 
                               xlabel='', ylabel='', 
                               title='', 
                               fontsize=13, 
                               save=False, 
                               savename='', 
                               ax0=None, ax1=None):
    if legend is None:
        legend = []


    created_figure = False
    if ax0 is None or ax1 is None:
        f, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        created_figure = True
    else:
        f = ax0.figure

    ax0.plot(xarr, true_signal[idx1], linestyle='solid', linewidth=2, color="#87dec4ff", label=legend[0])
    ax0.plot(xarr, predicted_signal[idx1], linestyle='dashed', linewidth=2, color="#15664eff")
    if idx2!=None:
        ax0.plot(xarr, true_signal[idx2], linestyle='solid', linewidth=2, color="#ffcc9cff", label=legend[1])
        ax0.plot(xarr, predicted_signal[idx2], linestyle='dashed', linewidth=2, color="#ff7b00ff")
    
    ax0.set_xscale('log')
    ax0.set_ylabel(ylabel, fontsize=fontsize)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    ax0.legend(title='True signal', fontsize=fontsize, title_fontsize=fontsize)
    if title:
        if created_figure:
            f.suptitle(title, fontsize=fontsize+2)
        else:
            ax0.set_title(title, fontsize=fontsize+2)

    ax1.axhline(0, linestyle='dashed', color='black', linewidth=1)
    ax1.plot(xarr, fractional_err[idx1], linestyle='solid', linewidth=2, color="#15664eff", label='fractional error')
    if idx2!=None:
        ax1.plot(xarr, fractional_err[idx2], linestyle='solid', linewidth=2, color="#ff7b00ff", label='fractional error')
    
    ax1.set_xscale('log')
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel('Fractional error', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize) 
    ax1.set_ylim(-0.1, 0.1)

    f.tight_layout()
    if save:
        plt.savefig('emu/figs/'+savename)
    elif created_figure:
        plt.show()

    return f, (ax0, ax1)


def plot_prediction_shapenoise(xarr, 
                               predicted_signal, 
                               true_signal, 
                               shape_noise,
                               l=0,
                               s=0,
                               idx1=0, idx2=None, 
                               legend=None, 
                               xlabel='', ylabel='', 
                               title='', 
                               fontsize=13, 
                               save=False, 
                               savename='', 
                               ax0=None, ax1=None):
    if legend is None:
        legend = []

    # get index corresponding to (l,s) from metadata    
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    pair_idx = metadata['bin_pairs'].index([l,s])

    created_figure = False
    if ax0 is None or ax1 is None:
        f, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        created_figure = True
    else:
        f = ax0.figure

    ax0.plot(xarr, true_signal[idx1]*1e3, linestyle='solid', linewidth=2, color="#87dec4ff", label=legend[0])
    ax0.plot(xarr, predicted_signal[idx1]*1e3, linestyle='dashed', linewidth=2, color="#15664eff")
    if idx2!=None:
        ax0.plot(xarr, true_signal[idx2]*1e3, linestyle='solid', linewidth=2, color="#ffcc9cff", label=legend[1])
        ax0.plot(xarr, predicted_signal[idx2]*1e3, linestyle='dashed', linewidth=2, color="#ff7b00ff")
    
    ax0.set_xscale('log')
    ax0.set_ylabel(ylabel, fontsize=fontsize)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    ax0.legend(title='True signal', fontsize=fontsize, title_fontsize=fontsize)
    if title:
        if created_figure:
            f.suptitle(title, fontsize=fontsize+2)
        else:
            ax0.set_title(title, fontsize=fontsize+2)

    ax1.axhline(0, linestyle='dashed', color='black', linewidth=1)
    ax1.plot(xarr, (predicted_signal[idx1] - true_signal[idx1]) / shape_noise[idx1][pair_idx], 
             linestyle='solid', 
             linewidth=2, 
             color="#15664eff", 
             label='fractional error')
    
    if idx2!=None:
        ax1.plot(xarr, (predicted_signal[idx2] - true_signal[idx2]) / shape_noise[idx2][pair_idx], 
                 linestyle='solid', 
                 linewidth=2, 
                 color="#ff7b00ff", 
                 label='fractional error')
    
    ax1.set_xscale('log')
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(r'$\sigma_e/\sqrt{N}$')  # --> stdev
    ax1.tick_params(axis='both', which='major', labelsize=fontsize) 
    # ax1.set_ylim(-0.1, 0.1)

    f.tight_layout()
    if save:
        plt.savefig('emu/figs/'+savename)
    elif created_figure:
        plt.show()

    return f, (ax0, ax1)