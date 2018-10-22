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

def plot_iv(I, V, idx, data_dir='data'):
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel('log current')
    plt.ylabel('voltage')
    plt.savefig(os.path.join(data_dir, 'iv_{}.png'.format(idx)))
    return

def run_cv_scan(cell='INTERNAL', verbose=False, initial_delay=30):
    """ run a CV scan for each point """

    # check on experiment status periodically:
    poll_interval = 1
    if verbose:
        print('initial delay', initial_delay)
    time.sleep(initial_delay)

    with asdc.control.controller(start_idx=17109013) as pstat:

        # run an open-circuit followed by a CV experiment
        status, oc_params = pstat.open_circuit(
            time_per_point=1, duration=60, current_range='AUTO', e_filter='1Hz', i_filter='1Hz'
        )

        if verbose:
            print('OC added.')
            print(status)
            print(oc_params)

        status, params = pstat.multi_cyclic_voltammetry(
            initial_potential=0.0, vertex_potential_1=-1.0, vertex_potential_2=1.0, final_potential=0.0, scan_rate=0.075,
            cell_to_use=cell, e_filter='1Hz', i_filter='1Hz', cycles=1
        )

        if verbose:
            print('CV added.')
            print(status)
            print(params)

        timestamp_start = datetime.now().isoformat(),
        pstat.start()

        error_codes = set()
        while pstat.sequence_running():
            time.sleep(poll_interval)
            overload_status = pstat.overload_status()
            if overload_status != 0:
                print('OVERLOAD:', overload_status)
                error_codes.add(overload_status)

        # collect and log data
        scan_data = {
            'measurement': 'cyclic_voltammetry',
            'parameters': params,
            'timestamp_start': timestamp_start,
            'timestamp': datetime.now().isoformat(),
            'current': pstat.current(),
            'potential': pstat.potential(),
            'error_codes': error_codes
        }

        pstat.clear()

    return scan_data

@click.command()
@click.argument('target-file', type=click.Path())
@click.option('--data-dir', default='data', type=click.Path())
@click.option('--delta-z', default=5e-5, type=float, help='z step in meters')
@click.option('--speed', default=1e-3, type=float, help='speed in meters/s')
@click.option('-c', '--cell', default='INTERNAL', type=click.Choice(['INTERNAL', 'EXTERNAL']))
@click.option('--verbose/--no-verbose', default=False)
def run_combi_scan(target_file, data_dir, delta_z, speed, cell, verbose):
    """ keep in mind that sample frame and versastat frame have x and y flipped:
    x_combi is -y_versastat
    y_combi is -x_versastat
    Also, combi wafer frame is in mm, versastat frame is in meters.
    Assume we start at the standard combi layout spot 1 (-9.04, -31.64)
    """

    df = pd.read_csv(target_file, index_col=0)

    current_spot = pd.Series(dict(x=-9.04, y=-31.64))

    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        initial_versastat_position = pos.current_position()

    for idx, target in df.iterrows():

        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        dy = -(target.x - current_spot.x) * 1e-3
        dx = -(target.y - current_spot.y) * 1e-3
        current_spot = target

        if verbose:
            print('position update:', dx, dy)

        with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
            delta = [dx, dy, 0.0]
            pos.update(delta=delta)
            current_v_position = pos.current_position()

        # run CV scan
        cv_data = run_cv_scan(cell, verbose=verbose, initial_delay=30)
        cv_data['index_in_sequence'] = idx
        cv_data['position_versa'] = current_v_position
        cv_data['position_combi'] = current_spot.to_dict()

        # log data
        logfile = 'grid_scan_{:03d}.json'.format(idx)
        with open(os.path.join(data_dir, logfile), 'w') as f:
            json.dump(cv_data, f)

        plot_iv(cv_data['current'], cv_data['potential'], idx, data_dir)

    # go back to the original position....
    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        x_initial, y_initial, z_initial = initial_versastat_position
        x_current, y_current, z_current = pos.current_position()
        delta = [x_initial - x_current, y_initial - y_current, 0.0]
        pos.update(delta=delta)

if __name__ == '__main__':
    run_combi_scan()
