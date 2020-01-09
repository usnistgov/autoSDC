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

import traceback

import cv2
from skimage import io

sys.path.append('../scirc')
sys.path.append('.')
import scirc

import asdc
from asdc import sdc
from asdc import slack
from asdc import visualization

asdc_channel = 'CDW5JFZAR'
BOT_TOKEN = open('slacktoken.txt', 'r').read().strip()
CTL_TOKEN = open('slack_bot_token.txt', 'r').read().strip()

def relative_flow(rates):
    """ convert a dictionary of flow rates to ratios of each component """
    total = sum(rates.values())
    if total == 0.0:
        return rates
    return {key: rate / total for key, rate in rates.items()}

class SDC(scirc.SlackClient):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, config=None, verbose=False, token=BOT_TOKEN, resume=False):
        super().__init__(verbose=verbose, logfile=config['logfile'].get(), token=token)
        self.command.update(super().command)
        self.msg_id = 0

        # load config values...

        # cell behavior
        self.cell = config['cell'].get()
        self.speed = config['speed'].get()
        self.stage_ip = config['stage_ip'].get()
        self.step_height = config['step_height'].get()
        self.compress_dz = config['compress_dz'].get()
        self.cleanup_pause = config['cleanup_pause'].get()

        self.solutions = config['solutions'].get()
        self.pump_array_port = config['pump_array_port'].get()
        self.adafruit_port = config['adafruit_port'].get()

        # which wafer direction is aligned with position controller +x direction?
        self.frame_orientation = config['frame_orientation'].get()

        # breakpoints and debugging options
        self.test = config['test'].get()
        self.test_cell = config['test_cell'].get()
        self.test_delay = config['test_delay'].get()
        self.confirm = config['confirm'].get()
        self.confirm_experiment = config['confirm_experiment'].get()

        # output options
        self.notify = config['notify_slack'].get()
        self.plot_cv = config['plot_cv'].get()
        self.plot_current = config['plot_current'].get()
        self.current_threshold = config['current_threshold'].get()

        # data serialization options
        self.data_dir = config['data_dir'].get()
        self.figure_dir = config['figure_dir'].get()

        self.db_file = os.path.join(self.data_dir, config['db_file'].get())
        self.db = dataset.connect(f'sqlite:///{self.db_file}')
        self.experiment_table = self.db['experiment']

        # initialize devices

        # coordinate systems positions
        self.initial_combi_position = pd.Series(config['initial_combi_position'].get())
        self.c_position = self.initial_combi_position

        with sdc.position.controller(ip=self.stage_ip) as pos:
            self.initial_versastat_position = pos.current_position()
            if self.verbose:
                print(f'initial vs position: {self.initial_versastat_position}')

        self.v_position = self.initial_versastat_position

        self.initialize_z_position = True
        self.resume = resume

        if self.resume:
            self.sync_coordinate_systems(register_initial=True)

        # syringe pump array
        try:
            self.pump_array = sdc.pump.PumpArray(
                self.solutions, port=self.pump_array_port, counterpump_port=self.adafruit_port
            )
        except:
            print('could not connect to pump array')
            self.pump_array = None

        # laser reflectance setup
        self.reflectometer = sdc.reflectivity.Reflectometer(port=self.adafruit_port)

    def sync_coordinate_systems(self, register_initial=False):

        with sdc.position.controller(ip=self.stage_ip) as pos:
            x_versa, y_versa = pos.x, pos.y

        # load last known combi position and update internal state accordingly
        refs = pd.DataFrame(self.experiment_table.all())
        if refs.size == 0:

            init = self.initial_combi_position
            ref = pd.Series({
                'x_versa': x_versa, 'y_versa': y_versa,
                'x_combi': init.x, 'y_combi': init.y
            })
        else:
            # arbitrarily grab the first position
            # TODO: verify that this record comes from the current session...
            ref = refs.iloc[0]

        # get the offset
        # convert versa -> combi (m -> mm)
        disp_x = (x_versa - ref.x_versa)*1e3
        disp_y = (y_versa - ref.y_versa)*1e3

        # keep track of the coordinate switch!
        if self.frame_orientation == '-y':
            # note: this one has been updated...
            # x_combi ~ -y_versa
            # y_combi ~ -x_versa
            x_combi = ref.x_combi - disp_y
            y_combi = ref.y_combi - disp_x
        elif self.frame_orientation == '-x':
            # x_vs is -x_c, y_vs is -y_c
            x_combi = ref.x_combi - disp_x
            y_combi = ref.y_combi - disp_y
        else:
            raise NotImplementedError

        self.c_position = pd.Series({'x': x_combi, 'y': y_combi})

        if register_initial:
            self.initial_combi_position = self.c_position
            if self.verbose:
                print(f"initial combi position: {self.c_position}")

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    def compute_position_update(self, dx, dy):
        """ map wafer frame position update to the position controller frame
        frame_orientation: which wafer direction is aligned with x_versastat?
        """

        if self.frame_orientation == '-y':
            # NOTE: this one has been updated.
            # default reference frame alignment
            # x_vs is -y_c, y_vs is -x_c
            delta = np.array([-dy, -dx, 0.0])

        elif self.frame_orientation == '-x':
            # x_vs is -x_c, y_vs is -y_c
            delta = np.array([-dx, -dy, 0.0])

        elif self.frame_orientation == '+y':
            # x_vs is y_c, y_vs is -x_c
            delta = np.array([dy, -dx, 0.0])

        elif self.frame_orientation == '+x':
            # x_vs is x_c, y_vs is y_c
            delta = np.array([dx, dy, 0.0])

        # convert from mm to m
        return delta * 1e-3

    @command
    async def move(self, ws, msgdata, args):

        args = json.loads(args)

        if self.verbose:
            print(args)

        self.sync_coordinate_systems()

        # specify target positions in combi reference frame
        # update this in the ctx manager even if things get cancelled...
        dx = args['x'] - self.c_position.x
        dy = args['y'] - self.c_position.y

        # map position update to position controller frame
        delta = self.compute_position_update(dx, dy)

        if (dx != 0) or (dy != 0):
            if self.verbose:
                print('position update: {} {} (mm)'.format(dx, dy))

            if self.notify:
                slack.post_message(f'*confirm update*: dx={dx}, dy={dy} (delta={delta})')

            async with sdc.position.acontroller(loop=self.loop, z_step=self.step_height, speed=self.speed) as pos:

                if self.confirm:
                    await ainput('press enter to allow lateral cell motion...', loop=self.loop)

                # move horizontally
                f = functools.partial(pos.update, delta=delta)
                await self.loop.run_in_executor(None, f)
                self.c_position += np.array([dx, dy])

            if self.verbose:
                print(pos.current_position())
                print(self.c_position)

        if self.initialize_z_position:
            # TODO: define the lower z baseline after the first move

            await ainput('*initialize z position*: press enter to continue...', loop=self.loop)
            self.initialize_z_position = False

        # @ctl -- update the semaphore in the controller process
        await self.dm_controller('<@UHNHM7198> update position is set.')

    async def set_flow(self, instruction, nominal_rate=0.5):
        """ nominal rate in ml/min """

        print('setting the flow rates directly!')
        params = f"rates={instruction.get('rates')} {instruction.get('units')}"
        hold_time = instruction.get('hold_time', 0)

        rates = instruction.get('rates')

        # if relative flow rates don't match, purge solution
        if relative_flow(rates) != relative_flow(self.pump_array.flow_setpoint):

            # high nominal flow_rate for running out to steady state
            total_rate = sum(rates.values())
            if total_rate <= 0.0:
                total_rate = 1.0

            line_flush_rates = {key: val * nominal_rate/total_rate for key, val in rates.items()}

            if self.notify:
                slack.post_message(f"flush lines at {line_flush_rates} ml/min")
                print('setting flow rates to flush the lines')

            self.pump_array.set_rates(line_flush_rates, counterpump_ratio='max')
            time.sleep(0.5)
            self.pump_array.run_all()

            print(f'waiting {hold_time} (s) for solution composition to reach steady state')
            time.sleep(hold_time)

        if self.notify:
            slack.post_message(f"set_flow to {rates} ml/min")
            print(f"setting flow rates to {rates} ml/min")

        # go to low nominal flow_rate for measurement
        self.pump_array.set_rates(rates)

    async def bump_flow(self, instruction, nominal_rate=0.5, duration=5):
        """ briefly increase the flow rate to make sure the cell gets filled

        TODO: maybe make this configurable by adding an argument to the set_flow op?
        """

        rates = instruction.get('rates')
        total_rate = sum(rates.values())
        cell_fill_rates = {key: val * nominal_rate/total_rate for key, val in rates.items()}

        if self.verbose:
            print(f"bump_flow to {cell_fill_rates} ml/min")

        self.pump_array.set_rates(cell_fill_rates, counterpump_ratio=0.75)
        time.sleep(0.5)
        self.pump_array.run_all()
        time.sleep(duration)
        self.pump_array.set_rates(rates)

    async def optical_inspect(self, x_combi=31.0, y_combi=0.0, delta_z=0.020):
        """ move for optical inspection
        delta_z should be specified in meters...
        """

        # make sure delta_z is positive (actually, greater than 500 microns...)
        delta_z = max(0.0005, delta_z)

        # specify target positions in combi reference frame
        print(x_combi, y_combi)
        print(self.c_position)
        dx = x_combi - self.c_position.x
        dy = y_combi - self.c_position.y

        # map position update to position controller frame
        delta = self.compute_position_update(dx, dy)

        print(delta)

        async with sdc.position.acontroller(loop=self.loop, z_step=self.step_height, speed=self.speed) as pos:

            if self.confirm:
                await ainput('press enter to move for optical inspection...', loop=self.loop)

            f = functools.partial(pos.update_z, delta=delta_z)
            await self.loop.run_in_executor(None, f)

            # move horizontally
            f = functools.partial(pos.update, delta=delta)
            await self.loop.run_in_executor(None, f)
            self.c_position += np.array([dx, dy])

        return

    @command
    async def run_experiment(self, ws, msgdata, args):
        """ run an SDC experiment """

        # args should contain a sequence of SDC experiments -- basically the "instructions"
        # segment of an autoprotocol protocol
        # that comply with the SDC experiment schema (TODO: finalize and enforce schema)
        instructions = json.loads(args)

        # check for an instruction group name/intent
        intent = instructions[0].get('intent')
        experiment_id = instructions[0].get('experiment_id')

        if intent is not None:
            instructions = instructions[1:]

        meta = {
            'intent': intent,
            'experiment_id': experiment_id,
            'instructions': json.dumps(instructions),
            'x_combi': float(self.c_position.x),
            'y_combi': float(self.c_position.y),
            'x_versa': self.v_position[0],
            'y_versa': self.v_position[1],
            'z_versa': self.v_position[2],
            'flag': False,
        }

        # wrap the whole experiment in a transaction
        # this way, if the experiment is cancelled, it's not committed to the db
        with self.db as tx:

            meta['id'] = tx['experiment'].insert(meta)

            stem = 'asdc'
            datafile = '{}_data_{:03d}.csv'.format(stem, meta['id'])

            if instructions[0].get('op') == 'set_flow':
                if self.test:
                    slack.post_message(f'we would set_flow here')
                else:
                    async with sdc.position.z_step(loop=self.loop, height=self.step_height, speed=self.speed):
                        await self.set_flow(instructions[0])

            # bump_flow needs to get run every time!
            await self.bump_flow(instructions[0], duration=10)

            summary = '-'.join(step['op'] for step in instructions)
            _msg = f"experiment *{meta['id']}*:  {summary}"

            if self.confirm_experiment:
                if self.notify:
                    slack.post_message(f'*confirm*: {_msg}')
                else:
                    print(f'*confirm*: {_msg}')
                await ainput('press enter to allow running the experiment...', loop=self.loop)

            elif self.notify:
                slack.post_message(_msg)

            # TODO: replace this with asyncio.run?
            f = functools.partial(
                sdc.experiment.run,
                instructions,
                cell=self.cell,
                verbose=self.verbose
            )
            if self.test_cell:
                slack.post_message(f"we would run the experiment here...")
                await self.loop.run_in_executor(None, time.sleep, 10)

            else:
                results, metadata = await self.loop.run_in_executor(None, f)

                metadata['parameters'] = json.dumps(metadata.get('parameters'))
                if self.pump_array:
                    metadata['flow_setpoint'] = json.dumps(self.pump_array.flow_setpoint)

                # TODO: define heuristic checks (and hard validation) as part of the experimental protocol API
                # heuristic check for experimental error signals?
                if np.median(np.abs(results['current'])) < self.current_threshold:
                    print(f'WARNING: median current below {self.current_threshold} threshold')
                    if self.notify:
                        slack.post_message(
                            f':terriblywrong: *something went wrong:*  median current below {self.current_threshold} threshold'
                        )

                meta.update(metadata)
                meta['datafile'] = datafile
                tx['experiment'].update(meta, ['id'])

                # store SDC results in external csv file
                results.to_csv(os.path.join(self.data_dir, datafile))

                if self.plot_current:
                    figpath = os.path.join(self.figure_dir, 'current_plot_{}.png'.format(meta['id']))
                    visualization.plot_i(results['elapsed_time'], results['current'], figpath=figpath)

                    if self.notify:
                        slack.post_message(f"finished experiment {meta['id']}: {summary}")
                        slack.post_image(figpath, title=f"current vs time {meta['id']}")

                if self.plot_cv:
                    figpath = os.path.join(self.figure_dir, 'cv_plot_{}.png'.format(meta['id']))
                    visualization.plot_cv(results['potential'], results['current'], segment=results['segment'], figpath=figpath)

                    if self.notify:
                        slack.post_message(f"finished experiment {meta['id']}: {summary}")
                        slack.post_image(figpath, title=f"CV {meta['id']}")

        if self.cleanup_pause > 0:

            async with sdc.position.z_step(loop=self.loop, height=self.step_height, speed=self.speed):
                self.pump_array.stop_all(counterbalance='full')
                time.sleep(self.cleanup_pause)
                self.pump_array.counterpump.stop()

        replicates = self.db['experiment'].count(experiment_id=experiment_id)
        if (intent == 'deposition') and (replicates == 2):

            if self.notify:
                slack.post_message(f"inspect deposit quality")

            inspection_dz = 0.020
            await self.optical_inspect(delta_z=inspection_dz)
            response = await ainput('take a moment to evaluate', loop=self.loop)

            async with sdc.position.acontroller(loop=self.loop, speed=self.speed) as pos:
                # drop back to baseline
                f = functools.partial(pos.update_z, delta=-inspection_dz)
                await self.loop.run_in_executor(None, f)

        await self.dm_controller('<@UHNHM7198> go')

    @command
    async def checkpoint(self, ws, msgdata, args):
        """ hold until user input is given to allow experiment to proceed """

        if self.notify:
            slack.post_message('*checkpoint reached*')

        await ainput('*checkpoint*: press enter to continue...', loop=self.loop)
        return await self.dm_controller('<@UHNHM7198> go')

    @command
    async def flag(self, ws, msgdata, args):
        """ mark a datapoint as bad
        TODO: format checking
        """
        primary_key = int(args)

        with self.db as tx:
            tx['experiment'].update({'id': primary_key, 'flag': True}, ['id'])

    @command
    async def coverage(self, ws, msgdata, args):
        """ record deposition coverage on (0.0,1.0). """
        primary_key, text = args.split(' ', 1)  # need to do format checking...
        primary_key = int(primary_key)
        coverage_estimate = float(text)

        if coverage_estimate < 0.0 or coverage_estimate > 1.0:
            slack.post_message(
                f':terriblywrong: *error:* coverage estimate should be in the range (0.0, 1.0)'
            )
        else:
            with self.db as tx:
                tx['experiment'].update({'id': primary_key, 'coverage': coverage_estimate}, ['id'])

    @command
    async def refl(self, ws, msgdata, args):
        """ record the reflectance of the deposit (0.0,inf). """
        primary_key, text = args.split(' ', 1)  # need to do format checking...
        primary_key = int(primary_key)
        reflectance_readout = float(text)

        if reflectance_readout < 0.0:
            slack.post_message(
                f':terriblywrong: *error:* reflectance readout should be positive'
            )
        else:
            with self.db as tx:
                tx['experiment'].update({'id': primary_key, 'reflectance': reflectance_readout}, ['id'])

    async def reflectance_linescan(self, stepsize=0.00015, n_steps=20):

        mean, var = [], []
        async with sdc.position.acontroller(loop=self.loop, speed=self.speed) as stage:

            for step in range(n_steps):

                reflectance_data = self.reflectometer.collect()
                mean.append(np.mean(reflectance_data))
                var.append(np.var(reflectance_data))

                stage.update_y(-stepsize)
                time.sleep(0.25)

        return mean, var

    @command
    async def reflectance(self, ws, msgdata, args):
        """ record the reflectance of the deposit (0.0,inf). """

        # get the stage position at the start of the linescan
        with sdc.position.controller() as stage:
            metadata = {'reflectance_xv': stage.x, 'reflectance_yv': stage.y}

        mean, var = await self.reflectance_linescan()

        if len(args) > 0:
            primary_key = int(args)
            filename = f'deposit_reflectance_{primary_key:03d}.json'

            metadata['id'] = primary_key
            metadata['reflectance_file'] = filename

            with self.db as tx:
                tx['experiment'].update(metadata, ['id'])

            with open(os.path.join(self.data_dir, filename), 'w') as f:
                json.dump(data, f)

        else:
            print(metadata)
            print(mean, var)

    @command
    async def imagecap(self, ws, msgdata, args):
        """ capture an image from the webcam.

        pass an experiment index to serialize metadata to db
        """

        camera = cv2.VideoCapture(1)
        status, frame = camera.read()
        camera.release()

        if len(args) > 0:
            primary_key = int(args)

            image_name = f'deposit_pic_{primary_key:03d}.png'

            with sdc.position.controller() as stage:
                metadata = {
                    'id': primary_key,
                    'image_xv': stage.x,
                    'image_yv': stage.y,
                    'image_name': image_name
                }

            with self.db as tx:
                tx['experiment'].update(metadata, ['id'])

        else:
            image_name = 'test-image.png'

        io.imsave(os.path.join(self.data_dir, image_name), frame)

    @command
    async def bubble(self, ws, msgdata, args):
        """ record a bubble in the deposit """
        primary_key = args  # need to do format checking...
        primary_key = int(primary_key)

        with self.db as tx:
            tx['experiment'].update({'id': primary_key, 'has_bubble': True}, ['id'])

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
    async def stop_pumps(self, ws, msgdata, args):
        """ shut off the syringe and counterbalance pumps """
        self.pump_array.stop_all(counterbalance='off')

    @command
    async def abort_running_handlers(self, ws, msgdata, args):
        """ cancel all currently running task handlers...

        WARNING: does not do any checks on the potentiostat -- don't call this while an experiment is running...

        we could register the coroutine address when we start it up, and broadcast that so it's cancellable...?
        """

        text = f"sdc: {msgdata['username']} said abort_running_handlers"
        print(text)

        # dm UC537488J (brian)
        response = await self.slack_api_call(
            'chat.postMessage',
            data={'channel': '<@UC537488J>', 'text': text, 'as_user': False, 'username': 'sdc'},
            token=CTL_TOKEN
        )
        return

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
@click.argument('experiment-root', type=click.Path())
@click.option('--resume/--no-resume', default=False)
@click.option('--verbose/--no-verbose', default=False)
def sdc_client(experiment_root, resume, verbose):

    config = asdc.config.OverrideConfig(experiment_root)

    # specify target file relative to config file
    config['target_file'] = os.path.join(experiment_root, config['target_file'].get())
    config['data_dir'] = os.path.join(experiment_root, config['data_dir'].get())
    config['figure_dir'] = os.path.join(experiment_root, config['figure_dir'].get())
    config['logfile'] = os.path.join(config['data_dir'].get(), config['command_logfile'].get())

    os.makedirs(config['data_dir'].get(), exist_ok=True)
    os.makedirs(config['figure_dir'].get(), exist_ok=True)

    # make sure step_height is positive!
    config['step_height'] = abs(config['step_height'].get())

    sdc = SDC(verbose=verbose, config=config, token=BOT_TOKEN, resume=resume)
    sdc.run()

if __name__ == '__main__':
    sdc_client()
