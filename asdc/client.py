import os
import sys
import json
import time
import click
import numpy as np
import pandas as pd
from ruamel import yaml

sys.path.append('../scirc')
import scirc

from asdc import sdc
from asdc import slack
from asdc import visualization

class SDC(scirc.Client):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, config=None, verbose=False):
        super().__init__(verbose=verbose)
        self.command.update(super().command)
        self.msg_id = 0

        with sdc.position.controller(ip='192.168.10.11') as pos:
            initial_versastat_position = pos.current_position()
            if self.verbose:
                print(f'initial vs position: {initial_versastat_position}')

        self.initial_versastat_position = initial_versastat_position
        self.initial_combi_position = pd.Series(config['initial_combi_position'])
        self.step_height = config.get('step_height', 0.0)
        self.compress_dz = config.get('compress_dz', 0.0)
        self.cell = config.get('cell', 'INTERNAL')
        self.speed = config.get('speed', 1e-3)
        self.data_dir = config.get('data_dir', os.getcwd())
        self.figure_dir = config.get('figure_dir', os.getcwd())
        self.confirm = config.get('confirm', True)

        self.v_position = self.initial_versastat_position
        self.c_position = self.initial_combi_position

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    @command
    async def move(self, ws, msgdata, args):
        print(args)
        args = json.loads(args)
        print(args['x'], args['y'])

        # specify target positions in combi reference frame
        dx = args['x'] - self.c_position.x
        dy = args['y'] - self.c_position.y
        # dx, dy = 1e-3, 1e-3
        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        delta = np.array([-dy, -dx, 0.0]) * 1e-3
        # current_spot = target

        if self.verbose:
            # print(current_spot.x, current_spot.y)
            print('position update: {} {} (mm)'.format(dx, dy))

        if self.confirm:
            await self.post(f'confirm update: dx={dx}, dy={dy} (delta={delta})', ws, msgdata)
            input('press enter to allow cell motion...')

        with sdc.position.controller(ip='192.168.10.11', speed=self.speed) as pos:
            if self.verbose:
                print(pos.current_position())

            pos.update(delta=delta, step_height=self.step_height, compress=self.compress_dz)
            self.v_position = pos.current_position()
            self.c_position += np.array([dx, dy])

            if self.verbose:
                print(pos.current_position())
                print(self.c_position)

        await self.post(f'moved dx={dx}, dy={dy} (delta={delta})', ws, msgdata['channel'])

    @command
    async def potentiostatic(self, ws, msgdata, args):
        print(args)
        args = json.loads(args)

        results = sdc.experiment.run_potentiostatic(args['potential'], args['duration'], cell=self.cell, verbose=self.verbose)
        time.sleep(args['duration'])

        results.update(args)
        results['position_versa'] = self.v_position
        results['position_combi'] = [float(self.c_position.x), float(self.c_position.y)]

        # log data
        idx = 0
        stem = 'test'
        logfile = '{}_data_{:03d}.json'.format(stem, idx)
        with open(os.path.join(self.data_dir, logfile), 'w') as f:
            json.dump(results, f)

        await self.post(
            f"finished potentiostatic scan V={args['potential']}, duration={args['duration']}"
            ws, msgdata['channel']
        )

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        await self.post(args, ws, dm_channel)

@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_client(config_file, verbose):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config['data_dir'] is None:
        config['data_dir'] = os.path.join(os.path.split(config_file)[0], 'data')

    if config['figure_dir'] is None:
        config['figure_dir'] = os.path.join(os.path.split(config_file)[0], 'figures')

    if config['step_height'] is not None:
        config['step_height'] = abs(config['step_height'])

    sdc = SDC(verbose=verbose, config=config)
    sdc.run()

if __name__ == '__main__':
    sdc_client()
