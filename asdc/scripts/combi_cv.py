import click
import time
import numpy as np
import pandas as pd

import asdc.position

def run_cv_scan():
    """ run a CV scan for each point """
    print('collect CV data here...')
    time.sleep(3)
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
        run_cv_scan()

    # go back to the original position....
    with asdc.position.controller(ip='192.168.10.11', speed=speed) as pos:
        x_initial, y_initial, z_initial = initial_versastat_position
        x_current, y_current, z_current = pos.current_position()
        delta = [x_initial - x_current, y_initial - y_current, 0.0]
        pos.update(delta=delta)

if __name__ == '__main__':
    run_combi_scan()
