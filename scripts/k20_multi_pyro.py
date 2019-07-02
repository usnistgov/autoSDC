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

from asdc import visualization
from asdc import analyze
from asdc import emulation

fig_dir = 'emulation/test'

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

def plot_emulator(model, domain, sample_posterior=True, fig_path=None):
    if sample_posterior:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,12))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

    features = ['V_oc', 'V_tp', 'I_p', 'slope']
    labels = [r'$V_{oc}$', r'$V_{tp}$', r'log $I_p$', 'slope']

    mean, var = model(domain)
    if sample_posterior:
        sample = model(domain, sample_posterior=True, noiseless=False)

    for col, feature, title in zip(axes.T, features, labels):
        m = model.models[feature]
        visualization.ternary_scatter_sub(m.X, m.y, ax=col[0])
        visualization.ternary_scatter_sub(domain.numpy(), mean[feature].numpy(), ax=col[1]);
        if sample_posterior:
            visualization.ternary_scatter_sub(domain.numpy(), sample[feature].numpy(), ax=col[2]);

        for ax in col:
            ax.axis('equal')
        col[0].set_title(title, size=18)

    plt.tight_layout()
    plt.subplots_adjust()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()

def sample_weights(alpha=alpha):
    """ sample objective function weighting from Dirichlet distribution
    specified by concentration parameters alpha
    """
    a = torch.tensor(list(alpha.values()), dtype=torch.float)
    weights = dist.Dirichlet(a).sample()
    return dict(zip(alpha.keys(), weights))

def random_scalarization_cb(model, candidates, cb_beta, weights=None, sign=sign):
    # scaling some data by some constant --> scaling the variance by the square of the constant

    objective = torch.zeros(candidates.size(0))

    with torch.no_grad():
        mean, var = model(candidates)

    for key, weight in weights.items():
        m = model.models[key]
        mu, v = mean[key], var[key]

        # remap objective function to [0,1]
        # use the observed data to set the scaling.
        min_val = (m.y * sign[key]).min()
        scale = torch.abs(m.y.max() - m.y.min())

        mu = mu * sign[key]
        mu = (mu - min_val) / scale
        v = v * (1/scale)**2

        sd = v.sqrt()
        ucb = mu + np.sqrt(cb_beta) * sd
        objective += weight * ucb

    return objective

def k20_optimize():
    """ multiobjective K20 problem. """
    db_file = 'data/k20-NiTiAl.db'
    targets = ['V_oc', 'I_p', 'V_tp', 'slope']
    domain = emulation.simplex_grid(30, buffer=0.05)
    D = torch.tensor(domain, dtype=torch.float)

    em = emulation.K20Wrapper(db_file, targets=targets, num_steps=100)
    print('ok')
    plot_emulator(em, D, fig_path=os.path.join(fig_dir, 'k20.png'))

    # start by sampling at the corners of the simplex
    # _s = D.argmax(0)
    _s = torch.randperm(D.size(0))[:10]
    X = D[_s]

    samples = [em.iter_sample(x) for x in X]
    Y_init = {
        key: torch.cat([s[key] for s in samples])
        for key in em.targets.keys()
    }

    model = emulation.ModelWrapper(
        pd.DataFrame(X.numpy(), columns=[em.inputs.keys()]),
        pd.DataFrame(Y_init),
        num_steps=100
    )
    plot_emulator(model, D, sample_posterior=False, fig_path=os.path.join(fig_dir, 'initial_model.png'))

    w = sample_weights()
    acq = random_scalarization_cb(model, D, weights=w, cb_beta=2)
    tax = visualization.ternary_scatter(D.numpy(), acq.numpy(), label='acquisition');
    tax.scatter(D[acq.argmax()].unsqueeze(0).numpy(), edgecolors='r', color='none', linewidths=2);
    plt.savefig(os.path.join(fig_dir, 'acquisition.png'), bbox_inches='tight')
    plt.clf(); plt.close()

    ## query the emulator and update the models
    x = D[acq.argmax()]
    y = em.iter_sample(x)

    for key, m in model.models.items():
        emulation.update_posterior(m, x_new=x[:-1], y_new=y[key])

    plot_emulator(model, D, sample_posterior=False, fig_path=os.path.join(fig_dir, 'updated_model.png'))

if __name__ == '__main__':
    k20_optimize()
