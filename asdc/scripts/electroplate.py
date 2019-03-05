#!/usr/bin/env python
import os
import sys
import glob
import json
import time
from ruamel import yaml
import click
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from asdc import sdc
from asdc import slack
from asdc import visualization

@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def electroplate(config_file, verbose):
    """ keep in mind that sample frame and versastat frame have x and y flipped:
    x_combi is -y_versastat
    y_combi is -x_versastat
    Also, combi wafer frame is in mm, versastat frame is in meters.
    Assume we start at the standard combi layout spot 1 (-9.04, -31.64)
    """

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config['data_dir'] is None:
        config['data_dir'] = os.path.split(config_file)[0]

    if config['delta_z'] is not None:
        config['delta_z'] = abs(config['delta_z'])

    df = pd.read_csv(config['target_file'], index_col=0)

    data_files = glob.glob(os.path.join(config['data_dir'], '*.json'))

    for composition_file in config['composition_file']:
        stem, ext = os.path.splitext(composition_file)
        lockfile = os.path.join(config['data_dir'], stem + '.lock')
        if not os.path.isfile(lockfile):
            break

    print('running depositions from ', composition_file)
    comp = pd.read_csv(os.path.join(config['data_dir'], composition_file), index_col=0)

    # add an initial dummy row for the CV...
    comp = pd.concat((comp.iloc[0:1], comp))
    comp.iloc[0] *= np.nan

    if len(data_files) == 0:
        current_spot = pd.Series(dict(x=-9.04, y=-31.64))
    else:
        current_spot = df.iloc[len(data_files)-1]
        df = df.iloc[len(data_files):]

    # check to see if we're starting in the middle of a composition...
    current_solution_datafiles = [
        data_file for data_file in data_files
        if stem in data_file
    ]
    if len(current_solution_datafiles) > 0:
        n_current = len(current_solution_datafiles)
        comp = comp.iloc[n_current:]
        run_cv = False
    else:
        run_cv = True

    print('start: ', current_spot.x, current_spot.y)

    if config['confirm']:
        # wait for user input to actually run
        input('press enter to run experiment')

    with sdc.position.controller(ip='192.168.10.11', speed=config['speed']) as pos:
        initial_versastat_position = pos.current_position()


    for (idx, target), (_, C) in zip(df.iterrows(), comp.iterrows()):

        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        dy = -(target.x - current_spot.x) * 1e-3
        dx = -(target.y - current_spot.y) * 1e-3
        delta = [dx, dy, 0.0]
        current_spot = target

        if verbose:
            print(current_spot.x, current_spot.y)
            # print('position update:', dx, dy)

        with sdc.position.controller(ip='192.168.10.11', speed=config['speed']) as pos:
            pos.update(delta=delta, step_height=config['delta_z'], compress=config['compress_dz'])
            current_v_position = pos.current_position()

        # run CV scan
        if run_cv:
            print('CV', current_spot.x, current_spot.y)

            if config['confirm']:
                input('press enter to run experiment')

            if config['initial_delay'] > 0:
                time.sleep(config['initial_delay']

            slack.post_message('Running a CV for {}.'.format(config['target_file']))
            the_data = sdc.experiment.run_cv_scan(cell=config['cell'], verbose=verbose)
            run_cv = False

            figpath = os.path.join(config['data_dir'], 'CV_{}.png'.format(idx))
            visualization.plot_vi(the_data['current'], the_data['potential'], figpath=figpath)
            slack.post_image(figpath, title='CV {}'.format(idx))

        else:
            potential = C['V']
            duration = C['t_100nm'] # time in seconds
            print('plate', C['f_Co'], 'Co')
            print('x={}, y={}, V={}, t={}'.format(current_spot.x, current_spot.y, potential, duration))

            if config['confirm']:
                input('press enter to run experiment')

            if config['initial_delay'] > 0:
                time.sleep(config['initial_delay']

            slack.post_message('Running electrodeposition targeting {} Co. ({}V for {}s)'.format(C['f_Co'], potential, duration))
            the_data = sdc.experiment.run_potentiostatic(potential, duration, cell=config['cell'], verbose=verbose)
            the_data.update(C.to_dict())

            figpath = os.path.join(config['data_dir'], 'current_plot_{}.png'.format(idx))
            visualization.plot_v(the_data['elapsed_time'], the_data['current'], figpath=figpath)
            slack.post_image(figpath, title='current vs time {}'.format(idx))


        the_data['index_in_sequence'] = int(idx)
        the_data['position_versa'] = current_v_position
        _spot = current_spot.to_dict()
        the_data['position_combi'] = [float(_spot['x']), float(_spot['y'])]

        # log data
        logfile = '{}_data_{:03d}.json'.format(stem, idx)
        with open(os.path.join(config['data_dir'], logfile), 'w') as f:
            json.dump(the_data, f)

    open(lockfile, 'a').close()

if __name__ == '__main__':
    electroplate()