#!/usr/bin/env python
""" Perform sparse measurements to estimate the location of the interface between the gold dot sample and the epoxy """

import GPy
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import os
import sys

import versastat.position
import versastat.control

# step size 100 microns
grid_step_size = 100 * 1e-6
grid_extent = 1e-3

INITIAL_Z_DELTA = 4.90e-4

n_measurements = 60
n_initial = 5

POTENTIAL_THRESHOLD = -0.8
CURRENT_THRESHOLD = '?'

KERNEL_LENGTHSCALE_CONSTRAINTS = (1e-4, 2e-3)

LOG_DIR = 'logs'
FIG_DIR = 'figures'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def acquire_cv_data(pstat, pos, idx):

    # run a CV experiment
    status, params = pstat.multi_cyclic_voltammetry(
        initial_potential=0.0, vertex_potential_1=-0.25, vertex_potential_2=0.65, final_potential=0.0, scan_rate=0.2,
        cell_to_use='EXTERNAL', e_filter='1Hz', i_filter='1Hz'
    )

    pstat.start()

    while pstat.sequence_running():
        time.sleep(poll_interval)

    # collect and log data
    scan_data = {
        'measurement': 'multi_cyclic_voltammetry',
        'parameters': params,
        'index_in_sequence': idx,
        'timestamp': datetime.now().isoformat(),
        'current': pstat.current(),
        'potential': pstat.potential(),
        'position': pos.current_position()
    }

    pstat.clear()

    return scan_data

def update_position(pos, new_xy):
    current_position = pos.current_position()
    z = current_position[-1]

    new_x, new_y = new_xy
    target_position = np.array([new_x, new_y, z])

    delta = target_position - current_position
    pos.update(delta, verbose=False)

def evaluate_CV_curves(potential, current, potential_threshold=POTENTIAL_THRESHOLD, current_threshold=CURRENT_THRESHOLD):
    """ make a polymer/metal decision by thresholding the current at the low end of the potenial curve """

    avg_current = np.mean(current[potential < potential_threshold])

    if avg_current < current_threshold:
        return 1
    else:
        return 0

def fit_gp(X, y, observed):

    m = GPy.models.GPClassification(
        X[observed],
        y[observed,None],
        kernel=GPy.kern.RBF(2, lengthscale=1.0) + GPy.kern.White(2, variance=0.05)
    )

    # NOTE: tune these lengthscales!
    m.kern.rbf.lengthscale.constrain_bounded(*KERNEL_LENGTHSCALE_CONSTRAINTS)
    m.kern.white.variance.constrain_bounded(1e-2, 0.1)
    m.optimize('bfgs', max_iters=100)

    mu, var = m.predict_noiseless(X)
    p = m.likelihood.gp_link.transf(mu)
    v = p - np.square(p)

    vv = v.copy()
    vv[observed] = 0
    query_id = np.argmax(vv)

    return m, p, v, query_id

def plot_predictions(X, y, p, var, query_id, idx):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    n_points, dim = X.shape
    s = int(np.sqrt(n_points))

    ax1.imshow(p.reshape((s,s)),
               origin='lower', aspect='equal', alpha=0.5, extent=extent, interpolation='bilinear', cmap='Reds')

    ax2.imshow(var.reshape((s,s)),
               origin='lower', aspect='equal', alpha=0.5, extent=extent, interpolation='bilinear', cmap='Reds')

    for ax in (ax1, ax2):
        ax.scatter(X[observed,0], X[observed,1], c=y[observed], edgecolors='k')
        ax.scatter(X[query_id,0], X[query_id,1], c='r', edgecolors='k')
        ax.contour(xx, yy, p.reshape((s,s)), levels=[0.5], colors='k', alpha=0.5, linestyles='dashed')
        ax.axis('off')

    plt.savefig(os.path.join(FIG_DIR, 'gp_predictions_{:03d}.png'.format(idx)))
    plt.clf()
    plt.close()

def plot_CV(potential, current, idx):
    plt.plot(potential, current, alpha=0.8)
    plt.savefig(os.path.join(FIG_DIR, 'CV_curve_{:03d}.png'.format(idx)))
    plt.clf()
    plt.close()

def write_logfile(scan_data, idx):
    """ serialize scan data to json """
    logfile = os.path.join(LOG_DIR, 'line_scan_{:03d}.json'.format(idx))
    with open(logfile, 'w') as f:
        json.dump(scan_data, f)

def find_circle(speed=1e-5, poll_interval=5):
    """ perform a sparse set of CV experiments, recording position, current, potential, and parameters in json log files
    Position units are METERS!
    """

    delta = [1e-4, 1e-4, 0.0]
    initial_delta = [0.0, 0.0, -INITIAL_Z_DELTA]
    final_delta = [0.0, 0.0, INITIAL_Z_DELTA]

    with versastat.position.controller(ip='192.168.10.11', speed=speed) as pos:
        pos.print_status()
        pos.update(delta=initial_delta, verbose=True)
        pos.print_status()

        # define a square measurement grid
        start_position = pos.current_position()
        xx, yy = np.meshgrid(np.arange(0,grid_extent,grid_step_size), np.arange(0,grid_extent,grid_step_size))
        X = np.c_[xx.ravel(), yy.ravel()]
        y = np.zeros(X.shape[0])
        observed = np.zeros_like(y, dtype=bool)

        with versastat.control.controller(start_idx=17109013) as pstat:
            pstat.set_current_range('20nA')

            # start in the lower left corner...
            query_id = 0
            for idx in range(n_steps):
                # train model, update position, scan, log,

                # run the experiment
                scan_data = acquire_cv_data(pstat, pos, idx)

                # make a decision
                label = evaluate_CV_curves(scan_data['potential'], scan_data['current'])
                scan_data['label'] = label
                y[query_id] = label
                observed[query_id] = True

                # save metadata
                write_logfile(scan_data, idx)

                # select the next point to measure
                if idx < n_initial:
                    # randomly select a point to query
                    query_id = np.random.choice(X.shape[0])
                else:
                    # actively acquire data
                    model, mu, var, query_id = fit_gp(X, y, observed)

                new_xy = X[query_id]

                # move the probe
                update_position(pos, new_xy)


                if idx < n_initial:
                    plot_predictions(X, p, var, query_id, idx)

                plot_CV(scan_data['potential'], scan_data['current'], idx)

                observed[query_id] = True

            # bring the probe back up
            pos.update(delta=final_delta, verbose=True)
            pos.print_status()


if __name__ == '__main__':
    find_circle()
