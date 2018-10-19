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


def plot_iv(I, V, idx):
    plt.plot(I, V)
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel('log current')
    plt.ylabel('voltage')
    plt.savefig('iv_{}.png'.format(idx))
    return

@click.group()
def cli():
    pass

@cli.command()
@click.option('-d', '--direction', default='x', type=click.Choice(['x', 'y', '+x', '-x', '+y', '-y']))
@click.option('--delta', default=5e-3, type=float, help='x step in meters')
@click.option('--delta-z', default=5e-5, type=float, help='z step in meters')
@click.option('--speed', default=1e-3, type=float, help='speed in meters/s')
@click.option('--verbose/--no-verbose', default=False)
def step(direction, delta, delta_z, speed, verbose):
    """ 1mm per second scan speed.
    up. over. down. down. up
    """

    # constrain absolute delta_z to avoid crashing....
    delta_z = np.clip(delta_z, -5e-5, 5e-5)

    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:

        # vertical step
        pos.update_z(delta=delta_z, verbose=verbose)

        # take the position step
        if verbose:
            pos.print_status()

        if 'x' in direction:
            update_position = pos.update_x
        elif 'y' in direction:
            update_position = pos.update_y

        if '-' in direction:
            delta *= -1

        update_position(delta=delta, verbose=verbose)

        if verbose:
            pos.print_status()

        # vertical step back down:
        # compress, then release
        pos.update_z(delta=-2*delta_z, verbose=verbose)
        pos.update_z(delta=delta_z, verbose=verbose)

        if verbose:
            pos.print_status()

@cli.command()
@click.option('--data-dir', default='data', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def cv(data_dir, verbose):
    """ run a CV experiment """

    # load previous datasets just to get current index...
    datafiles = glob.glob(os.path.join(data_dir, '*.json'))
    scan_idx = len(datafiles)
    with asdc.position.controller(ip='192.168.10.11') as pos:

        with asdc.control.controller(start_idx=17109013) as pstat:
            print('connected.')
            for idx in range(10):
                time.sleep(1)
                print('.', end='')
            print()
            # run an open-circuit followed by a CV experiment
            status, oc_params = pstat.open_circuit(
                time_per_point=1, duration=60, current_range='AUTO', e_filter='1Hz', i_filter='1Hz'
            )
            print('OC added.')
            if verbose:
                print(status)
                print(oc_params)
            status, params = pstat.multi_cyclic_voltammetry(
                initial_potential=0.0, vertex_potential_1=-1.0, vertex_potential_2=1.0, final_potential=0.0, scan_rate=0.075,
                cell_to_use='INTERNAL', e_filter='1Hz', i_filter='1Hz', cycles=1
            )
            print('CV added.')
            if verbose:
                print(status)
                print(params)

            print('starting.')
            pstat.start()

            while pstat.sequence_running():
                time.sleep(poll_interval)

            print('saving data')

            # collect and log data
            scan_data = {
                'measurement': 'cyclic_voltammetry',
                'parameters': params,
                'index_in_sequence': scan_idx,
                'timestamp': datetime.now().isoformat(),
                'current': pstat.current(),
                'potential': pstat.potential(),
                'position': pos.current_position()
            }

            logfile = 'grid_scan_{:03d}.json'.format(scan_idx)
            with open(os.path.join(data_dir, logfile), 'w') as f:
                json.dump(scan_data, f)

            plot_iv(scan_data['current'], scan_data['potential'], scan_idx)

            pstat.clear()

    return

if __name__ == '__main__':
    cli()
