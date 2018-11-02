import os
import numpy as np
import matplotlib.pyplot as plt

import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

def plot_iv(I, V, idx, data_dir='data'):
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel('log current')
    plt.ylabel('voltage')
    plt.savefig(os.path.join(data_dir, 'iv_{}.png'.format(idx)))
    plt.clf()
    plt.close()
    return

def plot_v(t, V, idx, data_dir='data'):
    plt.plot(np.arange(len(V)), V)
    plt.xlabel('time')
    plt.ylabel('voltage')
    plt.savefig(os.path.join(data_dir, 'v_{}.png'.format(idx)))
    plt.clf()
    plt.close()
    return
