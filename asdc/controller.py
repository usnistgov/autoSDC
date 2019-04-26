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

class Controller(scirc.Client):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()
    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN):
        super().__init__(verbose=verbose, logfile=logfile, token=token)
        self.command.update(super().command)
        self.msg_id = 0

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        # await self.post(args, ws, dm_channel)
        await self.api_call('chat.postMessage', data={'channel': dm_channel, 'text': args})


@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_controller(config_file, verbose):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config['data_dir'] is None:
        config['data_dir'] = os.path.join(os.path.split(config_file)[0], 'data')

    if config['figure_dir'] is None:
        config['figure_dir'] = os.path.join(os.path.split(config_file)[0], 'figures')

    if config['step_height'] is not None:
        config['step_height'] = abs(config['step_height'])

    # logfile = config.get('command_logfile', 'commands.log')
    logfile = 'controller.log'
    logfile = os.path.join(config['data_dir'], logfile)

    ctl = Controller(verbose=verbose, config=config, logfile=logfile)
    ctl.run()

if __name__ == '__main__':
    sdc_controller()
