import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import asdc.analyze

def plot_iv(I, V, figpath='iv.png'):
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel('log current')
    plt.ylabel('voltage')
    plt.savefig(figpath)
    plt.clf()
    plt.close()
    return

def plot_vi(I, V, figpath='iv.png'):
    plt.plot(V, np.log10(np.abs(I)))
    plt.ylabel('log current')
    plt.xlabel('voltage')
    plt.savefig(figpath)
    plt.clf()
    plt.close()
    return

def plot_v(t, V, figpath='v.png'):
    plt.plot(np.arange(len(V)), V)
    plt.xlabel('time')
    plt.ylabel('voltage')
    plt.savefig(figpath)
    plt.clf()
    plt.close()
    return

def plot_i(t, I, figpath='i.png'):
    plt.plot(np.arange(len(I)), I)
    plt.xlabel('time (s)')
    plt.ylabel('current (A)')
    plt.savefig(figpath)
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

def plot_open_circuit(current, potential, segment, figpath='open_circuit.png'):
    plt.figure(figsize=(4,5))

    model = asdc.analyze.extract_open_circuit_potential(current, potential, segment, return_model=True)
    plt.plot(-model.data, model.userkws['x'], color='b')
    plt.plot(-model.best_fit, model.userkws['x'], c='r', linestyle='--', alpha=0.5)
    plt.axhline(model.best_values['peak_loc'], c='k', linestyle='--', alpha=0.5)

    plt.xlabel('log current (log (A)')
    plt.ylabel('potential (V)')
    plt.savefig(figpath, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_ocp_model(x, y, ocp, gridpoints, model, query_position, figure_path=None):

    N, _ = gridpoints.shape
    w = int(np.sqrt(N))
    mu_y, var_y = model.predict_y(gridpoints)


    plt.figure(figsize=(5,4))
    combi_plot()

    plt.scatter(x,y, c=ocp, edgecolors='k', cmap='Blues')
    plt.axis('equal')

    cmap = plt.cm.Blues
    colors = Normalize(vmin=mu_y.min(), vmax=mu_y.max(), clip=True)(mu_y.flatten())
    # colors = mu_y.flatten()
    c = cmap(colors)
    a = Normalize(var_y.min(), var_y.max(), clip=True)(var_y.flatten())
    # c[...,-1] = 1-a

    c[np.sqrt(np.square(gridpoints).sum(axis=1)) > 76.2 / 2, -1] = 0
    c = c.reshape((w,w,4))

    extent = (np.min(gridpoints), np.max(gridpoints), np.min(gridpoints), np.max(gridpoints))
    im = plt.imshow(c, extent=extent, origin='lower', cmap=cmap);
    cbar = plt.colorbar(im, extend='both')
    plt.clim(mu_y.min(), mu_y.max())

    plt.scatter(query_position[0], query_position[1], c='none', edgecolors='r')
    plt.axis('equal')

    if figure_path is not None:
        plt.savefig(figure_path, bbox_inches='tight')
        plt.clf()
