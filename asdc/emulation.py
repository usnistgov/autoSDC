import torch
from torch import optim
import dataset
import numpy as np
import pandas as pd

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

def simplex_grid(n=3, buffer=0.1):
    """ construct a regular grid on the ternary simplex """

    xx, yy = np.meshgrid(np.linspace(0.0, 1., n), np.linspace(0.0, 1.0, n))
    s = np.c_[xx.flat,yy.flat]

    sel = np.abs(s).sum(axis=1) <= 1.0
    s = s[sel]
    ss = 1-s.sum(axis=1)
    s = np.hstack((s, ss[:,None]))

    scale = 1-(3*buffer)
    s = buffer + s*scale
    return s

def update_posterior(model, x_new=None, y_new=None, lr=1e-3, num_steps=150, optimize_noise_variance=True):

    if x_new is not None and y_new is not None:
        if x_new.ndimension() == 1:
            x_new = x_new.unsqueeze(0)
        X = torch.cat([model.X, x_new])
        # y = torch.cat([model.y, y_new.squeeze(1)])
        y = torch.cat([model.y, y_new])
        model.set_data(X, y)

    # update model noise prior based on variance of observed data
    model.set_prior('noise', dist.HalfNormal(model.y.var()))

    # reinitialize hyperparameters from prior
    p = model.kernel._priors
    model.kernel.variance = p['variance']()
    model.kernel.lengthscale = p['lengthscale'](model.kernel.lengthscale.size())


    if optimize_noise_variance:
        model.noise = model._priors['noise']()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam([param for name, param in model.named_parameters() if 'noise' not in name], lr=lr)

    losses = gp.util.train(model, optimizer, num_steps=num_steps)
    return losses

def model_ternary(X, y, drop_last=True, initial_noise_var=1e-4):
    """ set up GP model for single target """

    if drop_last:
        X = X[:,:-1] # ignore the last composition column

    sel = torch.isfinite(y)
    X, y = X[sel], y[sel]
    N, D = X.size()

    # set up ARD Matern 5/2 kernel
    # set an empirical mean function to the median value of observed data...
    kernel = gp.kernels.Matern52(input_dim=2, variance=torch.tensor(1.), lengthscale=torch.tensor([1.0, 1.0]))
    model = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(initial_noise_var))
    model.mean_function = lambda x: model.y.median()

    # set a weakly-informative lengthscale prior
    # e.g. half-normal(0, dx/3)
    dx = 1.0
    model.kernel.set_prior("lengthscale", dist.HalfNormal(dx/3))
    model.kernel.set_prior("variance", dist.Gamma(2.0, 1/2.0))

    # set a prior on the likelihood noise based on the variance of the observed data
    model.set_prior('noise', dist.HalfNormal(model.y.var()/2))

    return model

class ModelWrapper():

    def __init__(self, inputs, targets, optimize_noise_variance=True, dtype=torch.float, num_steps=2000):
        """ fit independent GP models for each target
        inputs X and targets should be data frames
        """

        self.dtype = dtype
        self.inputs = inputs
        self.targets = targets
        self.num_steps = num_steps
        self.optimize_noise_variance = optimize_noise_variance

        self.X = torch.tensor(inputs.values, dtype=dtype)
        self.Y = torch.tensor(targets.values, dtype=dtype)

        self.models = {}
        self.samplers = {}
        self.fit()

    def fit(self):

        for target in self.targets.keys():

            y = torch.tensor(self.targets[target].values, dtype=self.dtype)

            model = model_ternary(self.X, y, initial_noise_var=0.01)
            update_posterior(model, num_steps=self.num_steps, optimize_noise_variance=self.optimize_noise_variance)
            self.models[target] = model

    def __call__(self, X, target=None, noiseless=True, sample_posterior=False, n_samples=1, seed=None):
        """ evaluate GP models on compositions """

        if target is None:
            targets = self.models.keys()
        elif type(target) is str:
            targets = [target]

        mean, var = {}, {}
        for target in targets:
            model = self.models[target]

            with torch.no_grad():
                _mean, _cov = model(X[:,:-1], full_cov=True, noiseless=noiseless)

            if sample_posterior:

                n = _mean.size(0)
                jitter = torch.eye(n) * 1e-6

                L = torch.cholesky(_cov + jitter)
                V = torch.randn(X.size(0))

                mean[target] = _mean + L @ V

            else:
                mean[target] = _mean
                var[target] = _cov.diag()

        if sample_posterior:
            return mean
        else:
            return mean, var

    def iter_sample(self, X, target=None, noiseless=True):
        """ sample GP posteriors iteratively"""

        if X.ndimension() == 1:
            X = X.unsqueeze(0)

        if target is None:
            targets = self.models.keys()
        elif type(target) is str:
            targets = [target]

        mean = {}
        for target in targets:
            try:
                fn = self.samplers[target]
            except KeyError:
                model = self.models[target]
                fn = model.iter_sample(noiseless=noiseless)
                self.samplers[target] = fn

            mean[target] = fn(X[:,:-1])

        return mean


class K20Wrapper(ModelWrapper):
    """ Independent GPs fit to NiTiAl k20 dataset """
    def __init__(self, db_file, components=['Ni', 'Al', 'Ti'], targets=['V_oc', 'I_p', 'V_tp', 'slope', 'fwhm'], optimize_noise_variance=True, num_steps=2000):

        # load all the unflagged data from sqlite to pandas
        # use sqlite id as pandas index
        self.db = dataset.connect(f'sqlite:///{db_file}')
        self.df = pd.DataFrame(self.db['experiment'].all(flag=False))
        self.df.set_index('id', inplace=True)

        # # drop the anomalous point 45 that has a negative jog in the passivation...
        # self.df = self.df.drop(45)

        super().__init__(self.df.loc[:,components], self.df.loc[:,targets], num_steps=num_steps)


class K20v2Wrapper(ModelWrapper):
    """ Independent GPs fit to NiTiAl k20v2 dataset """
    def __init__(self, db_file, results_file, components=['Ni', 'Al', 'Ti'], targets=['V_oc', 'I_p', 'V_tp', 'slope'], optimize_noise_variance=True, num_steps=2000):

        # db_file = '../data/k20-NiTiAl-v2.db'
        # results_file = '../data/k20-NiTiAl-v2-results.db'
        self.db = dataset.connect(f'sqlite:///{db_file}')
        self.res = dataset.connect(f'sqlite:///{results_file}')
        experiment_table = self.db['experiment']

        # only load the more recent session...
        df = pd.DataFrame(experiment_table.find(session=2))
        r = pd.DataFrame(self.res['cycle0'].all())
        r = r[r['id'].isin(df['id'])]

        super().__init__(r.loc[:,components], r.loc[:,targets], num_steps=num_steps)
