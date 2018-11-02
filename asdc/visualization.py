import os
import numpy as np
import matplotlib.pyplot as plt

def plot_iv(I, V, idx, data_dir='data'):
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel('log current')
    plt.ylabel('voltage')
    plt.savefig(os.path.join(data_dir, 'iv_{}.png'.format(idx)))
    plt.clf()
    plt.close()
    return

def plot_v(V, data_dir='data'):
    plt.plot(V)
    plt.xlabel('time')
    plt.ylabel('voltage')
    plt.savefig(os.path.join(data_dir, 'v.png'))
    plt.clf()
    plt.close()
    return
