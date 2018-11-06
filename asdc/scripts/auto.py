#!/usr/bin/env python
import os
import sys
import glob
import json
import time
import click
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import asdc.control
import asdc.position
import asdc.experiment
import asdc.visualization
import asdc.analyze
import asdc.ocp
import asdc.slack

def update_position_combi(target, current_spot, speed=1e-3, delta_z=5e-3, compress=True, verbose=False):
    # update position: convert from mm to m
    # x_vs is -y_c, y_vs is x
    dy = -(target.x - current_spot.x) * 1e-3
    dx = -(target.y - current_spot.y) * 1e-3
    current_spot = target

    if verbose:
        print('position update:', dx, dy)

    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        # vertical step
        if delta_z > 0:
            pos.update_z(delta=delta_z)

        delta = [dx, dy, 0.0]
        pos.update(delta=delta)
        current_v_position = pos.current_position()

        if delta_z > 0:
            # vertical step back down:
            pos.update_z(delta=-delta_z)

        if compress:
            # compress 50 microns, then release
            compress_dz = 5e-5
            pos.update_z(delta=-compress_dz)
            pos.update_z(delta=compress_dz)

    return current_v_position

# 5mm up, 50 microns down
@click.command()
@click.argument('target-file', type=click.Path())
@click.option('-n', '--n-acquisitions', default=20, help='number of GP-driven measurements to collect')
@click.option('--data-dir', default='data', type=click.Path())
@click.option('--delta-z', default=5e-3, type=float, help='z step in meters')
@click.option('--speed', default=1e-3, type=float, help='speed in meters/s')
@click.option('-c', '--cell', default='INTERNAL', type=click.Choice(['INTERNAL', 'EXTERNAL']))
@click.option('--initial-delay', default=0, type=float, help='initial delay in s before running CV scan.')
@click.option('--lift/--no-lift', default=True, help='ease off vertically before horizontal motion.')
@click.option('--compress/--no-compress', default=True, help='press down below vertical setpoint to reseat probe after horizontal motion')
@click.option('--verbose/--no-verbose', default=False)
def run_auto_scan(target_file, n_acquisitions, data_dir, delta_z, speed, cell, initial_delay, lift, compress, verbose):
    """ collect three CV curves from predetermined spots, then use a Gaussian Process model to map out open circuit potential

    keep in mind that sample frame and versastat frame have x and y flipped:
    x_combi is -y_versastat
    y_combi is -x_versastat
    Also, combi wafer frame is in mm, versastat frame is in meters.
    Assume we start at the standard combi layout spot 1 (-9.04, -31.64)
    """

    delta_z = np.clip(delta_z, 0, 1e-1)
    if lift is False:
        delta_z = 0

    # assume we start at standard combi layout spot 1
    current_spot = pd.Series(dict(x=-9.04, y=-31.64))

    # get the corresponding versastat reference coordinates
    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        initial_versastat_position = pos.current_position()

    # kickstart with a few pre-determined scans...
    df = pd.read_csv(target_file, index_col=0)

    pre_collected_data = glob.glob(os.path.join(data_dir, '*.json'))
    offset = len(pre_collected_data)

    n_initial, _ = df.shape
    for idx in range(offset, offset + n_initial + n_acquisitions):

        if idx < n_initial:
            asdc.slack.post_message('acquiring predetermined spot {}'.format(idx))
            target = df.iloc[idx]
        else:
            asdc.slack.post_message('acquiring GP spot {}'.format(idx))
            target = asdc.ocp.gp_select(data_dir, plot_model=True)
            asdc.slack.post_image(os.path.join(data_dir, 'ocp_predictions_{}.png'.format(idx)), title='OCP map {}'.format(idx))

        # update position
        # specify target position in combi sample coordinates
        current_v_position = update_position_combi(target, current_spot, delta_z=delta_z, compress=compress, speed=speed)
        current_spot = target

        # run CV scan
        cv_data = asdc.experiment.run_cv_scan(cell, verbose=verbose, initial_delay=initial_delay)
        cv_data['index_in_sequence'] = int(idx)
        cv_data['position_versa'] = current_v_position
        _spot = current_spot.to_dict()
        cv_data['position_combi'] = [float(_spot['x']), float(_spot['y'])]

        # log data
        logfile = 'cv_scan_{:03d}.json'.format(idx)
        with open(os.path.join(data_dir, logfile), 'w') as f:
            json.dump(cv_data, f)

        figpath = os.path.join(data_dir, 'open_circuit_{}.png'.format(idx))
        asdc.visualization.plot_open_circuit(cv_data['current'], cv_data['potential'], cv_data['segment'], figpath=figpath)
        asdc.slack.post_image(figpath, title='open circuit {}'.format(idx))

        # asdc.visualization.plot_iv(cv_data['current'], cv_data['potential'], idx, data_dir)
        # asdc.visualization.plot_v(cv_data['elapsed_time'], cv_data['potential'], idx, data_dir=data_dir)

    # re-fit the GP after the final measurement
    target = asdc.ocp.gp_select(data_dir, plot_model=True)

    # go back to the original position....
    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        x_initial, y_initial, z_initial = initial_versastat_position
        x_current, y_current, z_current = pos.current_position()
        delta = [x_initial - x_current, y_initial - y_current, 0.0]

        # vertical step
        if lift:
            pos.update_z(delta=delta_z, verbose=verbose)

        pos.update(delta=delta)

        if lift:
            # vertical step back down:
            pos.update_z(delta=-delta_z, verbose=verbose)

        if compress:
            # compress 50 microns, then release
            compress_dz = 5e-5
            pos.update_z(delta=-compress_dz, verbose=verbose)
            pos.update_z(delta=compress_dz, verbose=verbose)


if __name__ == '__main__':
    run_auto_scan()
