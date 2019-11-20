""" GP deposition -- deposit duplicate samples, performing a corrosion test on the second """
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
from aioconsole import ainput

import matplotlib.pyplot as plt

import gpflow
import gpflowopt
from gpflowopt import acquisition
from scipy import stats
from scipy import spatial
from scipy import integrate
from datetime import datetime

sys.path.append('../scirc')
sys.path.append('.')
import scirc
import cycvolt
from asdc import slack
from asdc import analyze
from asdc import emulation
from asdc import visualization

import enum
Action = enum.Enum('Action', ['REPEAT', 'CORRODE', 'QUERY'])

BOT_TOKEN = open('slack_bot_token.txt', 'r').read().strip()
SDC_TOKEN = open('slacktoken.txt', 'r').read().strip()

def deposition_instructions(query, experiment_id=0):
    """ TODO: do something about deposition duration.... """
    instructions = [
        {
            "intent": "deposition",
            "experiment_id": experiment_id
        },
        {
            "op": "set_flow",
            "rates": {"CuSO4": query["flow_rate"]},
            "hold_time": 120
        },
        {
            "op": "potentiostatic",
            "potential": query["potential"],
            "duration": 300,
            "current_range": "20MA"
        }
    ]
    return instructions

def corrosion_instructions(experiment_id=0):
    instructions = [
        {
            "intent": "corrosion",
            "experiment_id": experiment_id
        },
        {
            "op": "set_flow",
            "rates": {"H2SO4": 0.1},
            "hold_time": 120
        },
        {
            "op": "potentiostatic",
            "potential": 0.1,
            "duration": 120,
            "current_range": "20MA"
        }
    ]
    return instructions

def exp_id(db):
    """ we're running two depositions followed by a corrosion experiment
    return 0 if it's time for the first deposition
           1 if it's time for  the second deposition
           2 if it's time for corrosion
    """
    deps = db['experiment'].count(intent='deposition')
    cors = db['experiment'].count(intent='corrosion')

    if cors == 0:
        phase = deps
    else:
        phase = deps % cors

    if phase in (0, 1):
        intent = 'deposition'
    else:
        intent = 'corrosion'
    if phase == 0:
        fit_gp = True
    else:
        fit_gp = False

    return intent, fit_gp

def select_action(db, threshold=0.9):
    """ run two depositions, followed by a corrosion experiment if the deposits are acceptable.
    """
    prev_id = db['experiment'].count()

    prev = db['experiment'].find_one(id=prev_id)

    if prev['intent'] == 'corrosion':
        return Action.QUERY

    elif prev['intent'] == 'deposition':
        n_repeats = db['experiment'].count(experiment_id=prev['experiment_id'])

        if n_repeats == 1:
            # logic to skip replicate based on quality goes here...
            return Action.REPEAT

        elif n_repeats == 2:
            # if coverage is good enough, run a corrosion measurement.
            session = pd.DataFrame(db['experiment'].find(experiment_id=prev['experiment_id']))
            min_coverage = session['coverage'].min()

            if min_coverage > threshold:
                target = db['experiment'].find_one(experiment_id=prev['experiment_id'], has_bubble=False)
                if target is None:
                    print('no replicates without bubbles...')
                    return Action.QUERY
                else:
                    print(f'good coverage ({min_coverage})')
                    print('target', target['id'])
                    pos = {'x': target['x_combi'], 'y': target['y_combi']}
                    return Action.CORRODE
            else:
                print(f'poor coverage ({min_coverage})')
                return Action.QUERY

def load_cv(row, data_dir='data', segment=2, half=True, log=True):
    """ load CV data and process it... """
    cv = pd.read_csv(os.path.join(data_dir, row['datafile']), index_col=0)

    sel = cv['segment'] == segment
    I = cv['current'][sel].values
    V = cv['potential'][sel].values
    t = cv['elapsed_time'][sel].values

    if half:
        # grab the length of the polarization curve
        n = I.size // 2
        I = I[:n]
        V = V[:n]

    if log:
        I = cycvolt.analyze.log_abs_current(I)

    return V, I, t - t[0]

def deposition_flow_rate(ins):
    i = json.loads(ins)
    try:
        return i[0]['rates']['CuSO4']
    except KeyError:
        return None

def deposition_potential(df):
    p = []
    for idx, row in df.iterrows():

        if row['intent'] == 'deposition':
            instructions = json.loads(row['instructions'])
            for instruction in json.loads(row['instructions']):
                if instruction.get('op') == 'potentiostatic':
                    p.append(instruction.get('potential'))
        elif row['intent'] == 'corrosion':
            p.append(None)
    return p

def load_experiment_files(csv_files, dir='.'):
    dir, _ = os.path.split(dir)
    file = os.path.join(dir, csv_file)
    if os.path.isfile(file):
        experiments = pd.concat(
            (pd.read_csv(file, index_col=0) for csv_file in csv_files),
            ignore_index=True
        )
    else:
        experiments = []
    return experiments

def load_experiment_json(experiment_files, dir='.'):
    """ an experiment file contains a json list of experiment definitions """
    dir, _ = os.path.split(dir)

    experiments = None
    for experiment_file in experiment_files:
        p = os.path.join(dir, experiment_file)
        if os.path.isfile(p):
            with open(p, 'r') as f:
                if experiments is None:
                    experiments = json.load(f)
                else:
                    experiments.append(json.load(f))
        else:
            experiments = []

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
        self.domain_file = config.get('domain_file')

        self.db_file = os.path.join(self.data_dir, config.get('db_file', 'test.db'))
        self.db = dataset.connect(f'sqlite:///{self.db_file}')
        self.experiment_table = self.db['experiment']

        self.targets = pd.read_csv(config['target_file'], index_col=0)
        self.experiments = load_experiment_json(config['experiment_file'], dir=self.data_dir)

        # remove experiments if there are records in the database
        num_prev = self.db['experiment'].count()
        if num_prev < len(self.experiments):
            self.experiments = self.experiments[num_prev:]
        else:
            self.experiments = []

        # gpflowopt minimizes objectives...
        # UCB switches to maximizing objectives...
        # swap signs for things we want to minimize (just integral current)
        self.objectives = ('integral_current', 'coverage')
        self.objective_alphas = [1,1]
        self.sgn = np.array([-1,1])

        # set up the optimization domain
        with open(os.path.join(self.data_dir, os.pardir, self.domain_file), 'r') as f:
            domain_data = json.load(f)

        # domain = None
        # for key, dim in domain_data['domain'].items():
        #     _d = gpflowopt.domain.ContinuousParameter(dim['name'], dim['min'], dim['max'])
        #     if domain is None:
        #         domain = _d
        #     else:
        #         domain += _d
        dmn = domain_data['domain']['x1']
        self.levels = [
            np.array([0.030, 0.050, 0.10, 0.30]),
            np.linspace(dmn['min'], dmn['max'], 20)
        ]
        self.ndim = [len(l) for l in self.levels][::-1]
        self.extent = [np.min(self.levels[0]), np.max(self.levels[0]), np.min(self.levels[1]), np.max(self.levels[1])]
        xx, yy = np.meshgrid(self.levels[0], self.levels[1])
        self.candidates = np.c_[xx.flatten(),yy.flatten()]

        # self.domain = domain

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
        # sqlite integer primary keys start at 1...
        df = pd.DataFrame(self.experiment_table.all())

        target_idx = self.experiment_table.count()
        experiment_idx = self.experiment_table.count(flag=False)

        return df, target_idx, experiment_idx

    def analyze_corrosion_features(self, segment=0):

        rtab = self.db.get_table('result', primary_id=False)

        for row in self.db['experiment'].all(intent='corrosion'):

            # extract features for any data that's missing
            if rtab.find_one(id=row['id']):
                continue

            d = {'id': row['id']}
            # V, log_I, t = load_cv(row, data_dir=self.data_dir, segment=segment)
            # cv_features, fit_data = cycvolt.analyze.model_polarization_curve(
            #     V, log_I, smooth=False, lm_method=None, shoulder_percentile=0.99
            # )

            # d.update(cv_features)
            # d['passive_region'] = d['V_tp'] - d['V_pass']

            V, I, t = load_cv(row, data_dir=self.data_dir, segment=segment, log=False, half=False)
            d['integral_current'] = np.abs(integrate.trapz(I, t))

            d['ts'] = datetime.now()
            rtab.upsert(d, ['id'])

        return

    def random_scalarization_cb(self, model_wrapper, candidates, cb_beta=0.25):
        """ random scalarization upper confidence bound acquisition policy function """

        objective = np.zeros(candidates.shape[0])

        # sample one set of weights from a dirichlet distribution
        # that specifies our general preference on the objective weightings
        weights = stats.dirichlet.rvs(self.objective_alphas).squeeze()

        if self.notify:
            slack.post_message(f'sampled objective fn weights: {weights}')

        for model, weight in zip(model_wrapper.models, weights):
            # set per-model confidence bound beta
            t = model.X.shape[0]
            cb_weight = cb_beta * np.log(2*t + 1)

            mean, var = model.predict_y(candidates)
            ucb = mean + cb_weight*np.sqrt(var)
            objective += weight * ucb.squeeze()

        return objective

    def gp_acquisition(self, resolution=100):

        if self.notify:
            slack.post_message(f'analyzing CV features...')

        # make sure all experiments are postprocessed and have values in the results table
        self.analyze_corrosion_features()

        if self.notify:
            slack.post_message(f'fitting GP models')

        # load positions, compositions, and measured values from db
        d = pd.DataFrame(self.db['experiment'].all())
        r = pd.DataFrame(self.db['result'].all())

        # get deposition metadata from instructions...
        d['flow_rate'] = d['instructions'].apply(deposition_flow_rate)
        d['potential'] = deposition_potential(d)
        d[['flow_rate', 'potential']] = d[['flow_rate','potential']].fillna(method='ffill')

        # split records into deposition and corrosion subsets...
        dep = d.loc[d['intent'] == 'deposition', ('id', 'flow_rate', 'potential', 'coverage')]
        cor = d.loc[d['intent'] == 'corrosion', ('id', 'flow_rate', 'potential')]
        cor = cor.merge(r, on='id')

        X_dep = dep.loc[:,('flow_rate', 'potential')].values
        Y_dep = dep['coverage'].values

        X_cor = cor.loc[:,('flow_rate', 'potential')].values
        Y_cor = cor['integral_current'].values

        # reset tf graph -- long-running program!
        gpflow.reset_default_graph_and_session()

        # set up models
        models = [
            emulation.model_synth(X_cor, Y_cor[:, 0][:,None], dx=0.25*np.ptp(self.candidates)),
            emulation.model_bounded(X_dep, Y_dep[:,None], dx=0.25*np.ptp(self.candidates))
        ]

        # set up multiobjective acquisition...
        # use this as a convenient model wrapper for now...
        model_wrapper = acquisition.HVProbabilityOfImprovement(models)

        for model in model_wrapper.models:
            # rescale model outputs to balance objectives...
            # skip this for now, for flexibility scaling things later...
            model.normalize_output = False
        model_wrapper.root._needs_setup = True
        model_wrapper.optimize_restarts = 1

        # fit the surrogate models
        # gpflowopt objective will optimize the full model list...
        model_wrapper._optimize_models()

        if self.notify:
            slack.post_message(f'evaluating acquisition function')

        # evaluate the acquisition function on a grid
        # acq = criterion.evaluate(candidates)
        acq = self.random_scalarization_cb(model_wrapper, self.candidates)

        # remove previously measured candidates
        mindist = spatial.distance.cdist(X_dep, self.candidates).min(axis=0)
        acq[mindist < 1e-5] = acq.min()

        # visualization.scatter_wafer(candidates*scale_factor, acq, label='acquisition', figpath=figpath)
        # if self.notify:
        #     slack.post_image(figpath, title=f"acquisition at t={t}")

        query_idx = np.argmax(acq)
        guess = self.candidates[query_idx]

        # plot the acquisition function...
        plt.figure(figsize=(4,4))
        figpath = os.path.join(self.figure_dir, f'acquisition_plot_{t}.png')
        extent = self.extent
        # extent = (self.domain.lower[0], self.domain.upper[0], self.domain.lower[1], self.domain.upper[1])
        plt.imshow(acq.reshape(self.ndim), cmap='Blues', extent=extent)
        plt.scatter(guess[0], guess[1], color='r')
        plt.scatter(X_dep[:,0], X_dep[:,1], color='k')
        plt.scatter(X_cor[:,0], X_cor[:,1], color='k')
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.xlabel('flow rate')
        plt.ylabel('potential')
        plt.tight_layout()
        plt.savefig(figpath, bbox_inches='tight')
        plt.clf()
        if self.notify:
            slack.post_image(figpath, title=f"acquisition at t={t}")

        query = pd.Series({'flow_rate': guess[0], 'potential': guess[1]})
        print(query)
        return query

    @command
    async def go(self, ws, msgdata, args):
        """ keep track of target positions and experiment list

        target and experiment indices start at 0
        sqlite integer primary keys start at 1...
        """

        previous_op = self.db['experiment'].find_one(id=self.db['experiment'].count())

        if len(self.experiments) > 0:
            instructions = self.experiments.pop(0)
            intent = instructions[0].get('intent')
            fit_gp = False
        else:
            instructions = None
            action, pos = select_action(self.db)
            # intent, fit_gp = exp_id(self.db)

        if action in {Action.QUERY, Action.REPEAT}:
            # march through target positions sequentially
            target_idx = self.db['experiment'].count(intent='deposition')
            target = self.targets.iloc[target_idx]
            pos = {'x': target.x, 'y': target.y}

        # if action is Action.CORRODE, select a target without a bubble to corrode
        if action == Action.CORRODE:
            target = self.db['experiment'].find_one(experiment_id=previous_op['experiment_id'], has_bubble=False)
            pos = {'x': target['x_combi'], 'y': target['y_combi']}

        # send the move command -- message @sdc
        self.update_event.clear()
        print(pos)
        await self.dm_sdc(f'<@UHT11TM6F> move {json.dumps(pos)}')
        print('waiting for ok')
        # wait for the ok -- @sdc will message us with `@ctl update position`...
        await self.update_event.wait()

        if instructions is None:

            if action in {Action.REPEAT, Action.CORRODE}:
                experiment_id = previous_op.get('experiment_id')
            else:
                # action == Action.QUERY
                experiment_id = previous_op.get('experiment_id') + 1

            print('get instructions')
            # get the next instruction set
            if action == Action.QUERY:
                query = self.gp_acquisition()
                instructions = deposition_instructions(query, experiment_id=experiment_id)
            elif action == Action.REPEAT:
                instructions = json.loads(previous_op['instructions'])
                instructions = instructions[1:] # skip the set_flow op...
                instructions = [{'intent': 'deposition', 'experiment_id': experiment_id}] + instructions
            elif action == Action.CORRODE:
                instructions = corrosion_instructions(experiment_id=experiment_id)

        print(instructions)
        # send the experiment command
        await self.dm_sdc(f"<@UHT11TM6F> run_experiment {json.dumps(instructions)}")

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

        response = await self.slack_api_call(
            'chat.postMessage', token=SDC_TOKEN,
            data={'channel': dm_channel, 'text': args, 'as_user': False, 'username': 'ctl'}
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


@click.command()
@click.argument('config-file', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
def sdc_controller(config_file, verbose):

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

    if config['step_height'] is not None:
        config['step_height'] = abs(config['step_height'])

    # logfile = config.get('command_logfile', 'commands.log')
    logfile = 'controller.log'
    logfile = os.path.join(config['data_dir'], logfile)

    ctl = Controller(verbose=verbose, config=config, logfile=logfile)
    ctl.run()

if __name__ == '__main__':
    sdc_controller()
