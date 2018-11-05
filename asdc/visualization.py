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

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))

def combi_plot():
    """ scatter plot visualizations on a 3-inch combi wafer.
    coordinate system is specified in mm
    """
    R = 76.2 / 2
    c = make_circle(R) # 3 inch wafer --> 76.2 mm diameter
    plt.plot(c[:,0], c[:,1], color='k')
