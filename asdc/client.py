import os
import sys
import json
from ruamel import yaml
import click

sys.path.append('../scirc')
import scirc

from asdc import sdc
from asdc import slack
# from asdc import visualization

class SDC(scirc.Client):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.command.update(super().command)

        with sdc.position.controller(ip='192.168.10.11') as pos:
            initial_versastat_position = pos.current_position()
            if self.verbose:
                print(f'initial vs position: {initial_versastat_position}')

    @command
    async def move(self, ws, msgdata, args):
        print(args)
        args = json.loads(args)
        print(args['x'], args['y'])
        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        # dy = -(args['x'] - current_spot.x) * 1e-3
        # dx = -(args['y'] - current_spot.y) * 1e-3
        dx, dy = 1e-3, 1e-3
        delta = [dx, dy, 0.0]
        # current_spot = target

        if self.verbose:
            # print(current_spot.x, current_spot.y)
            print('position update:', dx, dy)

        with sdc.position.controller(ip='192.168.10.11') as pos:
            # pos.update(delta=delta, step_height=config['delta_z'], compress=config['compress_dz'])
            pos.update(delta=delta)
            current_v_position = pos.current_position()

        r = f'moved dx={dx}, dy={dy}'
        response = {'id': 2, 'type': 'message', 'channel': msgdata['channel'], 'text': r}
        print(response)
        await ws.send_str(json.dumps(response))

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        response = {'id': 2, 'type': 'message', 'channel': dm_channel, 'text': args}
        await ws.send_str(json.dumps(response))

@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_client(config_file, verbose):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # if config['data_dir'] is None:
    #     config['data_dir'] = os.path.join(os.path.split(config_file)[0], 'data')

    # if config['figure_dir'] is None:
    #     config['figure_dir'] = os.path.join(os.path.split(config_file)[0], 'figures')

    # if config['delta_z'] is not None:
    #     config['delta_z'] = abs(config['delta_z'])

    sdc = SDC(verbose=verbose)
    sdc.run()

if __name__ == '__main__':
    sdc_client()
