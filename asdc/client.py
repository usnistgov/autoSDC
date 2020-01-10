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
import imageio

import sympy
from sympy.vector import express
from sympy.vector import CoordSys3D, BodyOrienter, Point

sys.path.append('../scirc')
sys.path.append('.')
import scirc

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

def to_vec(x, frame):
    """ convert python iterable coordinates to vector in specified reference frame """
    return x[0]*frame.i + x[1]*frame.j

def to_coords(x, frame):
    """ express coordinates in specified reference frame """
    return frame.origin.locate_new('P', to_vec(p, frame))

class SDC(scirc.SlackClient):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN, resume=False):
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
        self.cleanup_pause = config.get('cleanup_pause', 0)
        self.compress_dz = config.get('compress_dz', 0.0)
        self.cell = config.get('cell', 'INTERNAL')
        self.speed = config.get('speed', 1e-3)
        self.data_dir = config.get('data_dir', os.getcwd())
        self.figure_dir = config.get('figure_dir', os.getcwd())
        self.confirm = config.get('confirm', True)
        self.confirm_experiment = config.get('confirm_experiment', True)
        self.notify = config.get('notify_slack', True)
        self.plot_cv = config.get('plot_cv', False)
        self.plot_current = config.get('plot_current', False)

        self.test = config.get('test', False)
        self.test_cell = config.get('test_cell', False)
        self.test_delay = config.get('test', False)
        self.solutions = config.get('solutions')

        self.v_position = self.initial_versastat_position
        self.c_position = self.initial_combi_position

        self.initialize_z_position = True

        # which wafer direction is aligned with position controller +x direction?
        self.frame_orientation = config.get('frame_orientation', '-y')

        self.db_file = os.path.join(self.data_dir, config.get('db_file', 'test.db'))
        self.db = dataset.connect(f'sqlite:///{self.db_file}')
        self.experiment_table = self.db['experiment']

        self.current_threshold = 1e-5

        self.resume = resume

        # define reference frames
        # TODO: make camera and laser offsets configurable
        self.cell_frame = CoordSys3D('cell')
        self.camera_frame = self.cell_frame.locate_new('camera', 36*self.cell_frame.i)
        self.laser_frame = self.cell_frame.locate_new('laser', 48*self.cell_frame.i)

        if self.resume:
            self.stage_frame = self.sync_coordinate_systems(orientation=self.frame_orientation, register_initial=True, resume=self.resume)
        else:
            self.stage_frame = self.sync_coordinate_systems(orientation=self.frame_orientation, register_initial=False)

        adafruit_port = config.get('adafruit_port', 'COM9')
        pump_array_port = config.get('pump_array_port', 'COM10')

        try:
            self.pump_array = sdc.pump.PumpArray(
                self.solutions, port=pump_array_port, counterpump_port=adafruit_port
            )
        except:
            print('could not connect to pump array')
            self.pump_array = None

        self.reflectometer = sdc.reflectivity.Reflectometer(port=adafruit_port)

    def get_last_known_position(self, x_versa, y_versa, resume=False):

        # load last known combi position and update internal state accordingly
        refs = pd.DataFrame(self.experiment_table.all())

        if (resume == False) or (refs.size == 0):

            init = self.initial_combi_position
            ref = pd.Series({
                'x_versa': x_versa, 'y_versa': y_versa,
                'x_combi': init.x, 'y_combi': init.y
            })
        else:
            # arbitrarily grab the first position
            # TODO: verify that this record comes from the current session...
            ref = refs.iloc[0]

        return ref

    def sync_coordinate_systems(self, orientation=None, register_initial=False, resume=False):

        with sdc.position.controller() as pos:
            # map m -> mm
            x_versa = pos.x * 1e3
            y_versa = pos.y * 1e3

        ref = self.get_last_known_position(x_versa, y_versa, resume=resume)

        # set up the stage reference frame
        # relative to the last recorded positions
        cell = self.cell_frame

        if orientation == '-y':
            _stage = cell.orient_new('_stage', BodyOrienter(sympy.pi/2, sympy.pi, 0, 'ZYZ'))
        else:
            raise NotImplementedError

        # find the origin of the combi wafer in the coincident stage frame
        v = ref['x_combi']*cell.i + ref['y_combi']*cell.j
        combi_origin = v.to_matrix(_stage)

        # truncate to 2D vector
        combi_origin = np.array(combi_origin).squeeze()[:-1]

        # now find the origin of the stage frame
        xv_init = np.array([ref['x_versa'], ref['y_versa']])
        l = xv_init - combi_origin
        v_origin = l[1]*cell.i + l[0]*cell.j

        # construct the shifted stage frame
        stage = _stage.locate_new('stage', v_origin)
        return stage

    def compute_position_update(self, x, y, frame):
        """ compute frame update to map combi coordinate to the specified reference frame

        NOTE: all reference frames are in mm; the position controller works with meters
        """

        P = to_coords(p, frame)
        target_coords = np.array(P.express_coordinates(self.stage_frame))

        with sdc.position.controller() as pos:
            # map m -> mm
            current_coords = np.array((pos.x, pos.y, 0.0)) * 1e3

        delta = target_coords - current_coords

        # convert from mm to m
        return delta * 1e-3

    async def move_stage(self, x, y, frame, threshold=0.0001):
        """ specify target positions in combi reference frame
        threshold is specified in meters...
        only actually move if the update is above the noise floor
        """

        # map position update to position controller frame
        delta = self.compute_position_update(x, y, frame)

        if np.abs(delta.sum()) > threshold:
            if self.verbose:
                print(f'position update: {delta} (mm)')

            if self.notify:
                slack.post_message(f'*confirm update*: (delta={delta})')

            async with sdc.position.acontroller(loop=self.loop, z_step=self.step_height, speed=self.speed) as pos:

                if self.confirm:
                    await ainput('press enter to allow lateral cell motion...', loop=self.loop)

                # move horizontally
                f = functools.partial(pos.update, delta=delta)
                await self.loop.run_in_executor(None, f)

                if self.verbose:
                    print(pos.current_position())

        if self.initialize_z_position:
            # TODO: define the lower z baseline after the first move

            await ainput('*initialize z position*: press enter to continue...', loop=self.loop)
            self.initialize_z_position = False

        with sdc.position.controller() as pos:
            self.v_position = pos.current_position()

        return

    @command
    async def move(self, ws, msgdata, args):

        args = json.loads(args)

        if self.verbose:
            print(args)

        reference = args.get('reference_frame', 'cell')

        frame = {
            'cell': self.cell_frame,
            'laser': self.laser_frame,
            'camera': self.camera_frame
        }[reference]

        self.move_stage(args['x'], args['y'], frame)

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
            header = instructions[0]
            instructions = instructions[1:]

        # move now
        x_combi, y_combi = header.get('x'), header.get('y')
        self.move_stage(x_combi, y_combi, self.cell_frame)

        meta = {
            'intent': intent,
            'experiment_id': experiment_id,
            'instructions': json.dumps(instructions),
            'x_combi': float(x_combi),
            'y_combi': float(y_combi),
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
                mean.append(reflectance_data)
                # mean.append(np.mean(reflectance_data))
                # var.append(np.var(reflectance_data))

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
                data = {'reflectance': mean, 'variance': var}
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
        # give the camera enough time to come online before reading data...
        time.sleep(0.5)
        status, frame = camera.read()

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

        imageio.imsave(os.path.join(self.data_dir, image_name), frame)
        camera.release()

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

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {'id': self.msg_id, 'type': 'message', 'channel': channel, 'text': msg}
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

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
@click.argument('config-file', type=click.Path())
@click.option('--resume/--no-resume', default=False)
@click.option('--verbose/--no-verbose', default=False)
def sdc_client(config_file, resume, verbose):

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

    sdc = SDC(verbose=verbose, config=config, logfile=logfile, token=BOT_TOKEN, resume=resume)
    sdc.run()

if __name__ == '__main__':
    sdc_client()
