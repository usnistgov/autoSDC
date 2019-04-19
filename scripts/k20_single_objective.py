#!/usr/bin/env python

import os
import sys
import click
import gpflow
import numpy as np
import pandas as pd
from scipy import stats
from ruamel import yaml
import tensorflow as tf
from sklearn import metrics

import ternary
import matplotlib.pyplot as plt

# sys.path.append('..')
from asdc import analyze
from asdc import emulation
from asdc import visualization

opt = gpflow.training.ScipyOptimizer()

def confidence_bound(mu, var, sigma=2, minimize=True):
    """ confidence bound acquisition """
    if minimize:
        bound = -(mu.flat - sigma*np.sqrt(var.flat))
    else:
        bound = mu.flat + sigma*np.sqrt(var.flat)
    return bound

@click.command()
@click.argument('config-file', type=click.Path())
def k20_single_objective(config_file):
    """ optimize a single-objective function emulated by a GP fit to experimental data """

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_dir, _ = os.path.split(config_file)
    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(config)

    c = config['emulator']
    em = emulation.ExperimentEmulator(c['datafile'], components=c['components'])

    task = config['task']
    target = task['target']

    # set up a discrete grid of samples to optimize over...
    # randomize the grid because np.argmax takes the first value in memory order
    # if there are degenerate values
    domain = emulation.simplex_grid(task['domain_resolution'], buffer=task['buffer'])
    domain = domain[np.random.permutation(domain.shape[0])]

    visualization.ternary_scatter(domain, em(domain, target=target), label=target)
    plt.savefig(os.path.join(fig_dir, f'target_function_{target}.png'), bbox_inches='tight')
    plt.clf()

    # find max
    # s = emulation.simplex_grid(200, buffer=0.05)
    true_mu = em(domain, target=target)
    max_value = true_mu.max()
    print(max_value)

    # initialize
    queries = []
    s = emulation.simplex_grid(2, buffer=task['buffer'])
    v = em(s, target=target)

    mae, r2, ev = [], [], []
    for query_idx in range(task['budget']):

        # draw a picture
        visualization.ternary_scatter(s, v)
        plt.savefig(os.path.join(fig_dir, f'measured_{target}_{len(queries):02d}.png'), bbox_inches='tight')
        plt.clf()

        # fit the surrogate model
        m = emulation.model_ternary(s, v[:,None])
        opt.minimize(m)
        mu, var = m.predict_y(domain[:,:-1])

        # assess regret...
        print(f'query {query_idx}: {max_value - v.max()}')

        # evaluate predictive accuracy
        mae.append(np.mean(np.abs(true_mu - mu)))
        r2.append(metrics.r2_score(true_mu.flat, mu.flat))
        ev.append(metrics.explained_variance_score(true_mu.flat, mu.flat))
        print('MAE', mae[-1])
        print('R2', r2[-1])
        print('EV', ev[-1])
        plt.scatter(true_mu.flat, mu.flat)
        plt.plot((true_mu.min(), true_mu.max()), (true_mu.min(), true_mu.max()), linestyle='--', color='k')
        plt.savefig(os.path.join(fig_dir, f'parity_{target}_{len(queries):02d}.png'), bbox_inches='tight')
        plt.clf()

        # draw the extrapolations
        visualization.ternary_scatter(domain, mu.flat, label=target)
        plt.savefig(os.path.join(fig_dir, f'surrogate_{target}_{len(queries):02d}.png'), bbox_inches='tight')
        plt.clf()

        # acquisition = probability_of_improvement(mu, var, minimize=minimize)
        acquisition = confidence_bound(mu, var, sigma=task['kappa'], minimize=task['minimize'])
        acquisition[queries] = -np.inf

        visualization.ternary_scatter(domain, acquisition, label='acquisition')
        plt.savefig(os.path.join(fig_dir, f'acquisition_{target}_{len(queries):02d}.png'), bbox_inches='tight')
        plt.clf()

        # update the dataset
        queries.append(np.argmax(acquisition))
        query = domain[queries[-1]][None,:]
        s = np.vstack((s, query))
        v = np.hstack((v, em(query, target=target)))


    # draw a picture
    visualization.ternary_scatter(s, v)
    plt.savefig(os.path.join(fig_dir, f'measured_{target}_{len(queries)}.png'), bbox_inches='tight')
    plt.clf()

    plt.plot(3+np.arange(task['budget']), mae)
    plt.savefig(os.path.join(fig_dir, f'mae_{target}.png'), bbox_inches='tight')
    plt.clf()
    plt.plot(3+np.arange(task['budget']), r2)
    plt.savefig(os.path.join(fig_dir, f'R2_{target}.png'), bbox_inches='tight')
    plt.clf()
    plt.plot(3+np.arange(task['budget']), ev)
    plt.savefig(os.path.join(fig_dir, f'explained_variance_{target}.png'), bbox_inches='tight')
    plt.clf()

    return

if __name__ == '__main__':
    k20_single_objective()
