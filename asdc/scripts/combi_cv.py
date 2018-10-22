#!/usr/bin/env python
import os
import sys
import glob
import json
import time
import click
import numpy as np
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

def run_cv_scan(scan_idx, cell='INTERNAL', data_dir='data', verbose=False, initial_delay=30):
    """ run a CV scan for each point """

    # check on experiment status periodically:
    poll_interval = 1
    time.sleep(initial_delay)

    with asdc.position.controller(ip='192.168.10.11') as pos:

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

            pstat.start()

            error_codes = set()
            while pstat.sequence_running():
                time.sleep(poll_interval)
                overload_status = pstat.overload_status()
                if overload_status != 0:
                    error_codes.add(overload_status)

            # collect and log data
            scan_data = {
                'measurement': 'cyclic_voltammetry',
                'parameters': params,
                'index_in_sequence': scan_idx,
                'timestamp': datetime.now().isoformat(),
                'current': pstat.current(),
                'potential': pstat.potential(),
                'position': pos.current_position(),
                'error_codes': error_codes
            }

            logfile = 'grid_scan_{:03d}.json'.format(scan_idx)
            with open(os.path.join(data_dir, logfile), 'w') as f:
                json.dump(scan_data, f)

            plot_iv(scan_data['current'], scan_data['potential'], scan_idx, data_dir)

            pstat.clear()

    return

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
        print(current_spot)
        print(target)
        current_spot = target

        print(dx, dy)

        with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
            delta = [dx, dy, 0.0]
            pos.update(delta=delta)

        # run CV scan
        run_cv_scan(idx, cell, data_dir=data_dir, verbose=verbose, initial_delay=30)

    # go back to the original position....
    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        x_initial, y_initial, z_initial = initial_versastat_position
        x_current, y_current, z_current = pos.current_position()
        delta = [x_initial - x_current, y_initial - y_current, 0.0]
        pos.update(delta=delta)

if __name__ == '__main__':
    run_combi_scan()
