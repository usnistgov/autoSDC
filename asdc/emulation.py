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

def update_posterior(model, x_new=None, y_new=None, num_steps=150):

    if x_new is not None and y_new is not None:
        X = torch.cat([model.X, x_new])
        y = torch.cat([model.y, y_new.squeeze(1)])
        model.set_data(X, y)

    # update model noise prior based on variance of observed data
    model.set_prior('noise', dist.HalfNormal(model.y.var()))

    # reinitialize hyperparameters from prior
    p = model.kernel._priors
    model.kernel.variance = p['variance']()
    model.kernel.lengthscale = p['lengthscale'](model.kernel.lengthscale.size())
    # model.noise = model._priors['noise']()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    losses = gp.util.train(model, optimizer, num_steps=num_steps)
    return losses

def model_ternary(X, y, drop_last=True, optimize_noise_variance=True, initial_noise_var=1e-4):
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

    if not optimize_noise_variance:
        raise NotImplementedError
        for name, param in model.named_parameters():
            if name == 'noise_map_unconstrained':
                param.requires_grad = False

    return model

class ExperimentEmulator():
    def __init__(self, db_file, components=['Ni', 'Al', 'Ti'], targets = ['V_oc', 'I_p', 'V_tp', 'slope', 'fwhm'], optimize_noise_variance=True):
        """ fit independent GP models for each target -- read compositions and targets from a csv file... """

        # load all the unflagged data from sqlite to pandas
        # use sqlite id as pandas index
        self.db = dataset.connect(f'sqlite:///{db_file}')
        self.df = pd.DataFrame(self.db['experiment'].all(flag=False))
        self.df.set_index('id', inplace=True)

        # # drop the anomalous point 45 that has a negative jog in the passivation...
        # self.df = self.df.drop(45)

        self.components = components
        self.composition = self.df.loc[:,self.components].values
        self.targets = targets
        self.optimize_noise_variance = optimize_noise_variance

        self.models = {}
        self.samplers = {}
        self.fit()

    def fit(self):

        X = torch.tensor(self.composition, dtype=torch.float)

        for target in self.targets:
            y = torch.tensor(self.df[target].values, dtype=torch.float)

            model = model_ternary(X, y, initial_noise_var=0.1, optimize_noise_variance=self.optimize_noise_variance)
            update_posterior(model, num_steps=2000)
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

            mean[target] = _mean
            var[target] = _cov.diag()

        return mean, var

        if sample_posterior:
            if seed is not None:
                tf.set_random_seed(seed)
            mu = model.predict_f_samples(composition[:,:-1], n_samples)
            return mu.squeeze()
        else:
            mu, var = model.predict_y(composition[:,:-1])
            if return_var:
                return mu, var
            else:
                return mu.squeeze()

    def sample_posterior(self, X, target=None, noiseless=True):
        """ sample GP posteriors """

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
