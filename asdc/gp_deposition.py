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

            if min_coverage >= threshold:
                target = db['experiment'].find(experiment_id=prev['experiment_id'])
                target = target[~(target['has_bubble'] == True)]

                if target.shape[0] == 0:
                    print('no replicates without bubbles...')
                    return Action.QUERY
                else:
                    target = target.iloc[0]
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

def confidence_bound(model, candidates, sign=1, cb_beta=0.25):
    # set per-model confidence bound beta
    # default to lower confidence bound
    t = model.X.shape[0]
    cb_weight = cb_beta * np.log(2*t + 1)

    mean, var = model.predict_y(candidates)
    criterion = (sign*mean) - cb_weight*np.sqrt(var)
    return criterion

def classification_criterion(model, candidates):
    """ compute the classification criterion from 10.1007/s11263-009-0268-3 """
    loc, scale = model.predict_f(candidates)
    criterion = np.abs(loc) / np.sqrt(scale+0.001)
    return criterion

def plot_map(vals, X, guess, extent, figpath):
    plt.figure(figsize=(4,4))
    plt.imshow(vals, cmap='Blues', extent=extent, origin='lower')
    plt.colorbar()
    plt.scatter(X[:,0], X[:,1], color='k')
    plt.scatter(guess[0], guess[1], color='r')

    if 'coverage' in figpath:
        plt.contour(vals, levels=[0.5], extent=extent, colors='k', linestyles='--')

    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.xlabel('flow rate')
    plt.ylabel('potential')
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches='tight')
    plt.clf()

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
        self.coverage_threshold = config.get('coverage_threshold', 0.9)

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
        # classification criterion: minimization
        # confidence bound using LCB variant
        # swap signs for things we want to maximize (just coverage...)
        self.objectives = ('integral_current', 'coverage', 'reflectance')
        self.objective_alphas = [1,1,1]
        self.sgn = np.array([1, -1, -1])

        # set up the optimization domain
        with open(os.path.join(self.data_dir, os.pardir, self.domain_file), 'r') as f:
            domain_data = json.load(f)

        dmn = domain_data['domain']['x1']
        self.levels = [
            np.array([0.030, 0.050, 0.10, 0.30]),
            np.linspace(dmn['min'], dmn['max'], 50)
        ]
        # self.levels = [
        #     np.linspace(0.030, 0.30, 100),
        #     np.linspace(dmn['min'], dmn['max'], 100)
        # ]
        self.ndim = [len(l) for l in self.levels][::-1]
        self.extent = [np.min(self.levels[0]), np.max(self.levels[0]), np.min(self.levels[1]), np.max(self.levels[1])]
        xx, yy = np.meshgrid(self.levels[0], self.levels[1])
        self.candidates = np.c_[xx.flatten(),yy.flatten()]

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

    def random_scalarization_cb(self, models, candidates, cb_beta=0.25):
        """ random scalarization acquisition policy function
        depending on model likelihood, use different policy functions for different outputs
        each criterion should be framed as a minimization problem...
        """

        objective = np.zeros(candidates.shape[0])

        # sample one set of weights from a dirichlet distribution
        # that specifies our general preference on the objective weightings
        weights = stats.dirichlet.rvs(self.objective_alphas).squeeze()
        # weights = [0.0, 1.0]

        if self.notify:
            slack.post_message(f'sampled objective fn weights: {weights}')

        mask = None
        criteria = []
        for idx, model in enumerate(models):

            sign = self.sgn[idx]

            if model.likelihood.name in ('Gaussian', 'Beta'):
                criterion = confidence_bound(model, candidates, sign=sign, cb_beta=cb_beta)
            elif model.likelihood.name == 'Bernoulli':
                criterion = classification_criterion(model, candidates)
                y_loc, _ = model.predict_y(candidates)
                mask = (y_loc > 0.5).squeeze()

            criteria.append(criterion.squeeze())

        objective = np.zeros_like(criteria[0])
        for weight, criterion in zip(weights, criteria):
            if mask is not None:
                criterion[~mask] = np.inf
            drange = np.ptp(criterion[np.isfinite(criterion)])
            criterion = (criterion - criterion.min()) / drange
            objective += weight*criterion

        return objective

    def gp_acquisition(self, resolution=100, t=0):

        if self.notify:
            slack.post_message(f'analyzing CV features...')

        # make sure all experiments are postprocessed and have values in the results table
        self.analyze_corrosion_features()

        if self.notify:
            slack.post_message(f'fitting GP models')

        # load positions, compositions, and measured values from db
        d = pd.DataFrame(self.db['experiment'].all())
        print(d.columns)
        r = pd.DataFrame(self.db['result'].all())

        # get deposition metadata from instructions...
        d['flow_rate'] = d['instructions'].apply(deposition_flow_rate)
        d['potential'] = deposition_potential(d)
        d[['flow_rate', 'potential']] = d[['flow_rate','potential']].fillna(method='ffill')

        # split records into deposition and corrosion subsets...
        dep = d.loc[d['intent'] == 'deposition', ('id', 'experiment_id', 'flow_rate', 'potential', 'coverage', 'reflectance')]
        cor = d.loc[d['intent'] == 'corrosion', ('id', 'experiment_id', 'flow_rate', 'potential')]
        cor = cor.merge(r, on='id')

        # merge deposition quality into corrosion table...
        # drop any corrosion experiments where the coverage was below spec
        cor = cor.merge(d.loc[:,('experiment_id', 'coverage')].groupby('experiment_id').min(), on='experiment_id')
        cor = cor[cor['coverage'] > self.coverage_threshold]
        print(cor.shape)

        X_dep = dep.loc[:,('flow_rate', 'potential')].values
        Y_dep = (dep['coverage'].values > self.coverage_threshold).astype(float)

        X_cor = cor.loc[:,('flow_rate', 'potential')].values
        Y_cor = cor['integral_current'].values[:,None]

        # fit reflectance model only where coverage is good
        ref_selection = Y_dep == 1.0
        X_ref = X_dep[ref_selection]
        Y_ref = dep['reflectance'].values[ref_selection][:,None]

        # reset tf graph -- long-running program!
        gpflow.reset_default_graph_and_session()

        # set up models
        dx = 0.25*np.ptp(self.candidates)
        models = [
            emulation.model_property(X_cor, Y_cor[:, 0][:,None], dx=dx, optimize=True),
            emulation.model_quality(X_dep, Y_dep[:,None], dx=dx, likelihood='bernoulli', optimize=True),
            emulation.model_property(X_ref, Y_ref[:, 0][:,None], dx=dx, optimize=True),
        ]

        if self.notify:
            slack.post_message(f'evaluating acquisition function')

        # evaluate the acquisition function on a grid
        # acq = criterion.evaluate(candidates)
        acq = self.random_scalarization_cb(models, self.candidates)

        # remove previously measured candidates
        mindist = spatial.distance.cdist(X_dep, self.candidates).min(axis=0)
        acq[mindist < 1e-5] = np.inf

        # visualization.scatter_wafer(candidates*scale_factor, acq, label='acquisition', figpath=figpath)
        # if self.notify:
        #     slack.post_image(figpath, title=f"acquisition at t={t}")

        query_idx = np.argmin(acq)
        guess = self.candidates[query_idx]

        X = np.vstack((X_dep, X_cor))

        # plot the acquisition function...
        figpath = os.path.join(self.figure_dir, f'acquisition_plot_{t}.png')
        extent = self.extent
        plot_map(acq.reshape(self.ndim), X, guess, extent, figpath)

        if self.notify:
            slack.post_image(figpath, title=f"acquisition at t={t}")

        for objective, model in zip(self.objectives, models):
            loc, scale = model.predict_y(self.candidates)
            vals = loc.reshape(self.ndim)
            figpath = os.path.join(self.figure_dir, f'{objective}_plot_{t}.png')
            plot_map(vals, X, guess, extent, figpath)

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
            action = select_action(self.db, threshold=self.coverage_threshold)
            print(action)
            # intent, fit_gp = exp_id(self.db)


        if action in {Action.QUERY, Action.REPEAT}:
            # march through target positions sequentially
            target_idx = self.db['experiment'].count(intent='deposition')
            target = self.targets.iloc[target_idx]
            pos = {'x': target.x, 'y': target.y}

        # if action is Action.CORRODE, select a target without a bubble to corrode
        if action == Action.CORRODE:
            targets = self.db['experiment'].find(experiment_id=previous_op['experiment_id'])
            target = targets[~(targets['has_bubble'] == True)].iloc[0]

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
                query = self.gp_acquisition(t=experiment_id)
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
