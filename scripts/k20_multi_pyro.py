import os
import sys
import json
import yaml
import torch
import click
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
from asdc import k20_util
# from asdc.k20_util import plot_emulator

DTYPE = torch.double
torch.set_default_dtype(DTYPE)

def plot_acquisition(D, acq, fig_path=None):
    tax = visualization.ternary_scatter(D.numpy(), acq.numpy(), label='acquisition');
    tax.scatter(D[acq.argmax()].unsqueeze(0).numpy(), edgecolors='r', color='none', linewidths=2);
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf(); plt.close()


@click.command()
@click.argument('config-file', type=click.Path())
def k20_optimize(config_file):
    """ multiobjective K20 problem. """

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_dir, _ = os.path.split(config_file)
    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    targets = config['targets']
    sign = {k: np.sign(v) for k,v in targets.items()}
    alpha = {k: np.abs(v) for k,v in targets.items()}
    print(f'maximize {sign}')
    print(f'Dirichlet concentrations: {alpha}')

    domain = emulation.simplex_grid(30, buffer=0.05)
    D = torch.tensor(domain)
    plot_D = torch.tensor(emulation.simplex_grid(40, buffer=0.01))

    db_file = config['emulator']['datafile']
    if 'v2' in db_file:
        results_file = 'data/k20-NiTiAl-v2-results.db'
        em = emulation.K20v2Wrapper(db_file, results_file, targets=targets.keys(), num_steps=2000)
    else:
        em = emulation.K20Wrapper(db_file, targets=targets.keys(), num_steps=2000)

    k20_util.plot_emulator(em, plot_D, fig_path=os.path.join(fig_dir, 'k20_mean.png'))

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

    # plot the initial model
    k20_util.plot_emulator(model, plot_D, sample_posterior=False, fig_path=os.path.join(fig_dir, 'initial_model.png'))

    # also plot the noise-free posterior sample that defines the problem...
    posterior_grid, noise_grid = em.clean_iter_sample(plot_D, noiseless=False, uniform_noise=True)
    k20_util.plot_values(posterior_grid, plot_D, fig_path=os.path.join(fig_dir, 'k20_posterior_sample_noiseless.png'))
    k20_util.plot_values(
        {k: posterior_grid[k] + noise_grid[k] for k in posterior_grid.keys()},
        plot_D, fig_path=os.path.join(fig_dir, 'k20_posterior_sample.png')
    )
    k20_util.plot_pareto_set(posterior_grid, plot_D, sign=sign, fig_path=os.path.join(fig_dir, 'k20_posterior_pareto_set.png'))

    # mean, var = model(grid)

    for idx in range(config.get('budget', 10)):

        w = acquisition.sample_weights(alpha=alpha)
        cb_beta = 2
        # Samy's paper recommends 0.2dlog(2t)
        cb_beta = 0.2 * 2 * np.log(2*(idx+3) + 1)
        print(f'query {idx}: beta = {cb_beta}')
        acq = acquisition.random_scalarization_cb(model, D, weights=w, cb_beta=cb_beta, sign=sign)
        plot_acquisition(D, acq, fig_path=os.path.join(fig_dir, f'acquisition_{idx:02d}.png'))

        # remove previously measured candidates
        mindist = torch.cdist(model.models['I_p'].X, D[:,:-1]).min(1)
        acq[mindist.indices.unique()] = 0

        ## query the emulator and update the models
        x = D[acq.argmax()]

        # y = em.iter_sample(x)
        y, n = em.clean_iter_sample(x, uniform_noise=True)
        y_new = {key: y[key] + n[key] for key in y.keys()}

        print(f'x: {x}, y: {y}, ynew: {y_new}')
        for key, m in model.models.items():
            emulation.update_posterior(m, x_new=x[:-1], y_new=y_new[key])

        k20_util.plot_model_single(model, plot_D, fig_path=os.path.join(fig_dir, f'model_{idx:02d}.png'))
        k20_util.plot_model_paneled(model, plot_D, fig_path=os.path.join(fig_dir, f'model_{idx:02d}.png'))
        pred, _ = model(plot_D)
        k20_util.plot_pareto_set(pred, plot_D, sign=sign, fig_path=os.path.join(fig_dir, f'predicted_pareto_set_{idx:02d}.png'))
        k20_util.compare_pareto(
            pred, posterior_grid, plot_D, sign=sign, targets=['I_p', 'slope'],
            fig_path=os.path.join(fig_dir, f'pareto_front_comparison_{idx:02d}.png')
        )
        k20_util.plot_measured_pareto_set(model, em, fig_path = os.path.join(fig_dir, f'pareto_front_plot{idx:02d}.png'))



if __name__ == '__main__':
    k20_optimize()
