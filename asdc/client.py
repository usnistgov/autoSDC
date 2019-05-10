import os
import sys
import json
import time
import click
import asyncio
import dataset
import functools
import numpy as np
import pandas as pd
from ruamel import yaml
from datetime import datetime
from aioconsole import ainput, aprint
from contextlib import asynccontextmanager

sys.path.append('../scirc')
import scirc

from asdc import sdc
from asdc import slack
from asdc import visualization

asdc_channel = 'CDW5JFZAR'
BOT_TOKEN = open('slacktoken.txt', 'r').read().strip()
CTL_TOKEN = open('slack_bot_token.txt', 'r').read().strip()

class SDC(scirc.SlackClient):
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
        self.notify = config.get('notify_slack', True)
        self.test_delay = config.get('test', False)

        self.v_position = self.initial_versastat_position
        self.c_position = self.initial_combi_position

        self.db_file = os.path.join(self.data_dir, config.get('db_file', 'test.db'))
        self.db = dataset.connect(f'sqlite:///{self.db_file}')
        self.experiment_table = self.db['experiment']

        self.current_threshold = 1e-5


    @asynccontextmanager
    async def position_controller(self, use_z_step=False):
        """ wrap position controller context manager

        perform vertical steps before lateral cell motion with the ctx manager
        so that the cell drops back down to baseline z level if the `move` task is cancelled.

        Note: there seems to be some issue with cancelling an async task that uses position_controller...
        specifically, ainput seems to be causing some problems?
        Do we need to kill another task?
        https://stackoverflow.com/a/42294554

        Note: this doesn't seem to actually impact subsequent tasks -- there's just some error output.
        """
        _cancel_self_on_exit = False
        step = 0
        with sdc.position.controller(ip='192.168.10.11', speed=self.speed) as pos:
            start_position = pos.current_position()
            baseline_z = start_position[2]

            if self.verbose:
                c = pos.current_position()
                await aprint('1', c)

            try:
                if use_z_step > 0:
                    if self.confirm:
                        try:
                            await ainput('press enter to step up...', loop=self.loop)
                        except asyncio.CancelledError:
                            _cancel_self_on_exit = True
                            raise
                    f = functools.partial(pos.update_z, delta=self.step_height)
                    await self.loop.run_in_executor(None, f)

                yield pos

            finally:
                if use_z_step:
                    # go back to baseline
                    current_position = pos.current_position()
                    current_z = current_position[2]
                    dz = baseline_z - current_z

                    if self.confirm:
                        try:
                            await ainput('press enter to step back down...', loop=self.loop)
                        except asyncio.CancelledError:
                            _cancel_self_on_exit = True

                    f = functools.partial(pos.update_z, delta=dz)
                    await self.loop.run_in_executor(None, f)

                # update internal tracking of versastat position
                self.v_position = pos.current_position()
                if self.verbose:
                    c = pos.current_position()
                    await aprint('3', c)

                if _cancel_self_on_exit:
                    current_task = asyncio.current_task()
                    current_task.cancel()
                    raise asyncio.CancelledError

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    @command
    async def move(self, ws, msgdata, args):

        args = json.loads(args)

        if self.verbose:
            print(args)

        # specify target positions in combi reference frame
        # update this in the ctx manager even if things get cancelled...
        dx = args['x'] - self.c_position.x
        dy = args['y'] - self.c_position.y

        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        delta = np.array([-dy, -dx, 0.0]) * 1e-3

        if (dx != 0) or (dy != 0):
            if self.verbose:
                print('position update: {} {} (mm)'.format(dx, dy))

            if self.notify:
                slack.post_message(f'*confirm update*: dx={dx}, dy={dy} (delta={delta})')

            async with self.position_controller(use_z_step=True) as pos:

                if self.confirm:
                    await ainput('press enter to allow lateral cell motion...', loop=self.loop)

                # move horizontally
                f = functools.partial(pos.update, delta=delta)
                await self.loop.run_in_executor(None, f)
                self.c_position += np.array([dx, dy])

                if self.verbose:
                    await aprint('2', pos.current_position())

            if self.verbose:
                print(pos.current_position())
                print(self.c_position)

        # @ctl
        await self.dm_controller('<@UHNHM7198> update position is set.')
        time.sleep(1)
        if self.notify:
            slack.post_message(f'moved dx={dx}, dy={dy} (delta={delta})')

    @command
    async def potentiostatic(self, ws, msgdata, args):
        """ run a potentiostatic experiment (for e.g. electrodeposition) """

        args = json.loads(args)
        args['x_combi'] = float(self.c_position.x)
        args['y_combi'] = float(self.c_position.y)
        args['x_versa'] = self.v_position[0]
        args['y_versa'] = self.v_position[1]
        args['z_versa'] = self.v_position[2]
        args['flag'] = False
        args['comment'] = ''

        # wrap the whole experiment in a transaction
        # this way, if the experiment is cancelled, it's not committed to the db
        with self.db as tx:
            args['id'] = tx['experiment'].insert(args)

            stem = 'test'
            datafile = '{}_data_{:03d}.json'.format(stem, args['id'])

            _msg = "potentiostatic scan *{id}*:  V={potential}, t={duration}".format(**args)
            if self.confirm:
                if self.notify:
                    slack.post_message(f'*confirm*: {_msg}')
                else:
                    print(f'*confirm*: {_msg}')
                await ainput('press enter to allow running the experiment...', loop=self.loop)

            elif self.notify:
                slack.post_message(_msg)

            # TODO: replace this with asyncio.run?
            # results = sdc.experiment.run_potentiostatic(args['potential'], args['duration'], cell=self.cell, verbose=self.verbose)
            f = functools.partial(
                sdc.experiment.run_potentiostatic,
                args['potential'], args['duration'], pre_potential=-0.5, pre_duration=10,
                cell=self.cell, verbose=self.verbose
            )
            results, metadata = await self.loop.run_in_executor(None, f)
            metadata['parameters'] = json.dumps(metadata['parameters'])
            if self.test_delay:
                await self.loop.run_in_executor(None, time.sleep, 10)

            # heuristic check for experimental error signals?
            if np.median(np.abs(results['current'])) < self.current_threshold:
                print(f'WARNING: median current below {self.current_threshold} threshold')
                if self.notify:
                    slack.post_message(f':terriblywrong: *something went wrong:*  median current below {self.current_threshold} threshold')

            args.update(metadata)
            args['results'] = json.dumps(results)

            tx['experiment'].update(args, ['id'])

        # dump data
        with open(os.path.join(self.data_dir, datafile), 'w') as f:
            for key, value in args.items():
                if type(value) == datetime:
                    args[key] = value.isoformat()
            json.dump(args, f)

        figpath = os.path.join(self.figure_dir, 'current_plot_{}.png'.format(args['id']))
        visualization.plot_i(results['elapsed_time'], results['current'], figpath=figpath)

        if self.notify:
            slack.post_message(f"finished potentiostatic scan V={args['potential']}, duration={args['duration']}")
            time.sleep(1)

            # post an image
            slack.post_image(figpath, title='current vs time {}'.format(args['id']))
            time.sleep(1)

        await self.dm_controller('<@UHNHM7198> go')

    @command
    async def flag(self, ws, msgdata, args):
        """ mark a datapoint as bad
        TODO: format checking
        """
        primary_key = int(args)

        with self.db as tx:
            tx['experiment'].update({'id': primary_key, 'flag': True}, ['id'])

    @command
    async def comment(self, ws, msgdata, args):
        """ add a comment """
        primary_key, text = args.split(' ', 1)  # need to do format checking...
        primary_key = int(primary_key)

        row = self.experiment_table.find_one(id=primary_key)

        if row['comment']:
            comment = row['comment']
            comment += '; '
            comment += text
        else:
            comment = text

        with self.db as tx:
            tx['experiment'].update({'id': primary_key, 'comment': comment}, ['id'])

    async def dm_controller(self, text, channel='DHNHM74TU'):
        response = await self.slack_api_call(
            'chat.postMessage',
            data={'channel': channel, 'text': text, 'as_user': False, 'username': 'sdc'},
            token=CTL_TOKEN
        )

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        print('got a dm command: ', args)
        response = await self.slack_api_call(
            'chat.postMessage',
            data={'channel': dm_channel, 'text': args, 'as_user': False, 'username': 'sdc'},
            token=CTL_TOKEN
        )

    @command
    async def abort_running_handlers(self, ws, msgdata, args):
        """ cancel all currently running task handlers...

        WARNING: does not do any checks on the potentiostat -- don't call this while an experiment is running...

        we could register the coroutine address when we start it up, and broadcast that so it's cancellable...?
        """
        current_task = asyncio.current_task()

        for task in asyncio.all_tasks():

            if task._coro == current_task._coro:
                continue

            if task._coro.__name__ == 'handle':
                print(f'killing task {task._coro}')
                task.cancel()

        # ask the controller to cancel the caller task too...!
        await self.dm_controller('<@UHNHM7198> abort_running_handlers')


@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_client(config_file, verbose):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    experiment_root, _ = os.path.split(config_file)

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

    # make sure step_height is positive!
    if config['step_height'] is not None:
        config['step_height'] = abs(config['step_height'])

    logfile = config.get('command_logfile', 'commands.log')
    logfile = os.path.join(config['data_dir'], logfile)

    sdc = SDC(verbose=verbose, config=config, logfile=logfile, token=BOT_TOKEN)
    sdc.run()

if __name__ == '__main__':
    sdc_client()