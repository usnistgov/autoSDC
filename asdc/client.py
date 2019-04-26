import os
import sys
import json
import time
import click
import asyncio
import functools
import numpy as np
import pandas as pd
from ruamel import yaml
from aioconsole import ainput

sys.path.append('../scirc')
import scirc

from asdc import sdc
from asdc import slack
from asdc import visualization

BOT_TOKEN = open('slacktoken.txt', 'r').read().strip()
CTL_TOKEN = open('slack_bot_token.txt', 'r').read().strip()

class SDC(scirc.Client):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN):
        super().__init__(verbose=verbose, logfile=logfile, token=token)
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

        self.pandas_file = os.path.join(self.data_dir, config.get('pandas_file', 'test.csv'))

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
            await self.post(f'*confirm update*: dx={dx}, dy={dy} (delta={delta})', ws, msgdata['channel'])
            await ainput('press enter to allow cell motion...')

        with sdc.position.controller(ip='192.168.10.11', speed=self.speed) as pos:
            if self.verbose:
                print(pos.current_position())

            # pos.update(delta=delta, step_height=self.step_height, compress=self.compress_dz)
            f = functools.partial(pos.update, delta=delta, step_height=self.step_height, compress=self.compress_dz)
            await self.loop.run_in_executor(None, f)

            self.v_position = pos.current_position()
            self.c_position += np.array([dx, dy])

            if self.verbose:
                print(pos.current_position())
                print(self.c_position)

        await self.dm_controller('i finished the thing.')
        time.sleep(1)
        await self.post(f'moved dx={dx}, dy={dy} (delta={delta})', ws, msgdata['channel'])

    @command
    async def potentiostatic(self, ws, msgdata, args):
        # TODO: database/pandas load/store routine
        # TODO: implement flag and comment handlers
        print(args)
        args = json.loads(args)

        try:
            df = pd.read_csv(self.pandas_file, index_col=0)
            idx = df.index.max() + 1
        except:
            df = None
            idx = 0

        stem = 'test'
        logfile = '{}_data_{:03d}.json'.format(stem, idx)

        _msg = f"potentiostatic scan {idx}:  V={args['potential']}, t={args['duration']}"
        if self.confirm:
            await self.post(f'*confirm*: {_msg}', ws, msgdata['channel'])
            await ainput('press enter to allow run the experiment...')
        else:
            await self.post(_msg, ws, msgdata['channel'])

        # results = sdc.experiment.run_potentiostatic(args['potential'], args['duration'], cell=self.cell, verbose=self.verbose)
        f = functools.partial(sdc.experiment.run_potentiostatic, args['potential'], args['duration'], cell=self.cell, verbose=self.verbose)
        results = await self.loop.run_in_executor(None, f)

        await self.loop.run_in_executor(None, time.sleep, args['duration'])

        results.update(args)
        results['position_versa'] = self.v_position
        results['position_combi'] = [float(self.c_position.x), float(self.c_position.y)]
        results['flag'] = False
        results['comment'] = ''

        # log data
        with open(os.path.join(self.data_dir, logfile), 'w') as f:
            json.dump(results, f)

        _df = pd.DataFrame.from_dict(results, orient='index').T
        if df is not None:
            df = pd.concat((df, _df), ignore_index=True)
        else:
            df = _df

        df.to_csv(self.pandas_file)

        await self.post(
            f"finished potentiostatic scan V={args['potential']}, duration={args['duration']}",
            ws, msgdata['channel']
        )

    @command
    async def flag(self, ws, msgdata, args):
        """ mark a datapoint as bad """
        idx = int(args) # need to do format checking...

        df = pd.read_csv(self.pandas_file, index_col=0)
        df.at[idx, 'flag'] = True
        df.to_csv(self.pandas_file)

    @command
    async def comment(self, ws, msgdata, args):
        """ add a comment """
        idx, text = args.split(' ', 1)  # need to do format checking...
        idx = int(idx)

        df = pd.read_csv(self.pandas_file, index_col=0)
        df['comment'] = df.comment.fillna('')

        if df.at[idx, 'comment']:
            df.at[idx, 'comment'] += '; '
            df.at[idx, 'comment'] += text
        else:
            df.at[idx, 'comment'] = text

        df.to_csv(self.pandas_file)

    async def dm_controller(self, text, channel='DHNHM74TU'):
        response = await self.api_call(
            'chat.postMessage',
            data={'channel': channel, 'text': text, 'as_user': False, 'username': 'sdc'},
            token=CTL_TOKEN
        )

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        print('got a dm command: ', args)
        response = await self.api_call(
            'chat.postMessage',
            data={'channel': dm_channel, 'text': args, 'as_user': False, 'username': 'sdc'},
            token=CTL_TOKEN
        )

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

    logfile = config.get('command_logfile', 'commands.log')
    logfile = os.path.join(config['data_dir'], logfile)

    sdc = SDC(verbose=verbose, config=config, logfile=logfile, token=BOT_TOKEN)
    sdc.run()

if __name__ == '__main__':
    sdc_client()
