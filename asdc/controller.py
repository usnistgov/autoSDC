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

from asdc import slack

BOT_TOKEN = open('slack_bot_token.txt', 'r').read().strip()
SDC_TOKEN = open('slacktoken.txt', 'r').read().strip()

def load_experiment_files(csv_files, dir='.'):
    dir, _ = os.path.split(dir)
    experiments = pd.concat(
        (pd.read_csv(os.path.join(dir, csv_file), index_col=0) for csv_file in csv_files),
        ignore_index=True
    )
    return experiments

class Controller(scirc.SlackClient):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()
    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN):
        super().__init__(verbose=verbose, logfile=logfile, token=token)
        self.command.update(super().command)
        self.msg_id = 0
        self.update_event = asyncio.Event(loop=self.loop)

        self.confirm = config.get('confirm', True)
        self.notify = config.get('notify_slack', True)
        self.data_dir = config.get('data_dir', os.getcwd())
        self.figure_dir = config.get('figure_dir', os.getcwd())

        self.pandas_file = os.path.join(self.data_dir, config.get('pandas_file', 'test.csv'))
        self.targets = pd.read_csv(config['target_file'], index_col=0)
        self.experiments = load_experiment_files(config['composition_file'], dir=self.data_dir)

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    async def dm_sdc(self, text, channel='DHY5REQ0H'):
        response = await self.slack_api_call(
            'chat.postMessage',
            data={'channel': channel, 'text': text, 'as_user': False, 'username': 'ctl'},
            token=SDC_TOKEN
        )

    def load_experiment_indices(self):
        # indices start at 0...
        if os.path.isfile(self.pandas_file):
            df = pd.read_csv(self.pandas_file)
            experiment_idx = df[~df['flag']].shape[0]
            target_idx = df.shape[0]
        else:
            df = None
            experiment_idx = 0
            target_idx = 0

        return df, target_idx, experiment_idx

    @command
    async def go(self, ws, msgdata, args):
        df, target_idx, experiment_idx = self.load_experiment_indices()
        print(experiment_idx, target_idx)

        target = self.targets.iloc[target_idx]
        experiment = self.experiments.iloc[experiment_idx]
        print(target)
        print(experiment)

        # send the move command...
        # message @sdc
        self.update_event.clear()
        args = {'x': target.x, 'y': target.y}
        await self.dm_sdc(f'<@UHT11TM6F> move {json.dumps(args)}')

        # wait for the ok
        # @sdc will message us with @ctl update position ...
        await self.update_event.wait()

        # the move was successful and we've had our chance to the previous spot
        # reload the experiment in case flags have changed
        df, target_idx, experiment_idx = self.load_experiment_indices()
        experiment = self.experiments.iloc[experiment_idx]
        print(experiment)

        # send the experiment command
        args = {'potential': experiment['V'], 'duration': experiment['t_100nm']}
        await self.dm_sdc(f'<@UHT11TM6F> potentiostatic {json.dumps(args)}')

        return

    @command
    async def update(self, ws, msgdata, args):
        update_type, rest = args.split(' ', 1)
        print(update_type)
        self.update_event.set()
        return

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHY5REQ0H'
        # dm_channel = 'DHNHM74TU'
        # await self.post(args, ws, dm_channel)
        response = await self.slack_api_call(
            'chat.postMessage', token=SDC_TOKEN,
            data={'channel': dm_channel, 'text': args, 'as_user': False, 'username': 'ctl'}
        )
        # print(response)


@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_controller(config_file, verbose):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    experiment_root, _ = os.path.split(config_file)[0]

    # specify target file relative to config file
    target_file = config.get('target_file')
    config['target_file'] = os.path.join(experiment_root, target_file)

    data_dir = config.get('data_dir')
    if data_dir is None:
        config['data_dir'] = os.path.join(experiment_root, 'data')

    figure_dir = config.get('figure_dir')
    if figure_dir is None:
        config['figure_dir'] = os.path.join(experiment_root, 'figures')

    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['figure_dir'], exist_ok=True)

    if config['step_height'] is not None:
        config['step_height'] = abs(config['step_height'])

    # logfile = config.get('command_logfile', 'commands.log')
    logfile = 'controller.log'
    logfile = os.path.join(config['data_dir'], logfile)

    ctl = Controller(verbose=verbose, config=config, logfile=logfile)
    ctl.run()

if __name__ == '__main__':
    sdc_controller()
