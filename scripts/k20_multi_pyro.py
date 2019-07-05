import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn import linear_model
from gpflowopt.pareto import non_dominated_sort

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import matplotlib.pyplot as plt

from asdc import analyze
from asdc import emulation
from asdc import acquisition
from asdc import visualization
from asdc.k20_util import plot_emulator

fig_dir = 'emulation/test'
DTYPE = torch.double
torch.set_default_dtype(DTYPE)

# set dirichlet distribution concentration parameters
# for each objective
alpha = {
    'I_p': 3,
    'slope': 3,
    'V_oc': 2,
    'V_tp': 1
}

# maximization problem:
# e.g. maximize the negative passivation current...
sign = {
    'I_p': -1,
    'slope': -1,
    'V_oc': -1,
    'V_tp': 1
}

def plot_acquisition(D, acq, fig_path=None):
    tax = visualization.ternary_scatter(D.numpy(), acq.numpy(), label='acquisition');
    tax.scatter(D[acq.argmax()].unsqueeze(0).numpy(), edgecolors='r', color='none', linewidths=2);
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf(); plt.close()

def k20_optimize():
    """ multiobjective K20 problem. """
    budget = 20
    # db_file = 'data/k20-NiTiAl.db'
    db_file = 'data/k20-NiTiAl-v2.db'
    results_file = 'data/k20-NiTiAl-v2-results.db'
    targets = ['V_oc', 'I_p', 'V_tp', 'slope']
    domain = emulation.simplex_grid(50, buffer=0.01)
    D = torch.tensor(domain)

    em = emulation.K20v2Wrapper(db_file, results_file, targets=targets, num_steps=2000)
    print('ok')
    plot_emulator(em, D, fig_path=os.path.join(fig_dir, 'k20.png'))

    # start by sampling at the corners of the simplex
    # _s = torch.randperm(D.size(0))[:10]
    _s = D.argmax(0)
    X = D[_s]

    # samples = [em.iter_sample(x) for x in X]
    # Y_init = {
    #     key: torch.cat([s[key] for s in samples])
    #     for key in em.targets.keys()
    # }
    sample_init, noise_init = em.clean_iter_sample(X, noiseless=False, uniform_noise=True)
    Y_init = {key: (sample_init[key] + noise_init[key]) for key in sample_init.keys()}

    model = emulation.ModelWrapper(
        pd.DataFrame(X.numpy(), columns=[em.inputs.keys()]),
        pd.DataFrame(Y_init),
        num_steps=250
    )
    plot_emulator(model, D, sample_posterior=False, fig_path=os.path.join(fig_dir, 'initial_model.png'))

    for idx in range(budget):
        w = acquisition.sample_weights(alpha=alpha)
        acq = acquisition.random_scalarization_cb(model, D, weights=w, cb_beta=1, sign=sign)
        plot_acquisition(D, acq, fig_path=os.path.join(fig_dir, f'acquisition_{idx:02d}.png'))

        ## query the emulator and update the models
        x = D[acq.argmax()]
        # y = em.iter_sample(x)
        y, n = em.clean_iter_sample(x, uniform_noise=True)
        y_new = {key: y[key] + n[key] for key in y.keys()}

        for key, m in model.models.items():
            emulation.update_posterior(m, x_new=x[:-1], y_new=y_new[key])

        plot_emulator(model, D, sample_posterior=False, fig_path=os.path.join(fig_dir, f'model_{idx:02d}.png'))

if __name__ == '__main__':
    k20_optimize()
