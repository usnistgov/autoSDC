import torch
from torch import optim
import dataset
import numpy as np
import pandas as pd

import pyro
import pyro.contrib.gp as gp
from pyro.contrib.gp.util import conditional
import pyro.distributions as dist

DTYPE = torch.double
torch.set_default_dtype(DTYPE)

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

def mod_iter_sample(model, noiseless=True):
    r"""
        Iteratively constructs a sample from the Gaussian Process posterior.
        Recall that at test input points :math:`X_{new}`, the posterior is
        multivariate Gaussian distributed with mean and covariance matrix
        given by :func:`forward`.
        This method samples lazily from this multivariate Gaussian. The advantage
        of this approach is that later query points can depend upon earlier ones.
        Particularly useful when the querying is to be done by an optimisation
        routine.
        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).
        :param bool noiseless: A flag to decide if we want to add sampling noise
            to the samples beyond the noise inherent in the GP posterior.
        :returns: sampler
        :rtype: function
    """
    noise = model.noise.detach()
    X = model.X.clone().detach()
    y = model.y.clone().detach()
    N = X.size(0)
    Kff = model.kernel(X).contiguous()
    if not noiseless:
        Kff.view(-1)[::N + 1] += noise  # add noise to the diagonal
    else:
        Kff.view(-1)[::N + 1] += 1e-8  # add noise to the diagonal

    outside_vars = {"X": X, "y": y, "N": N, "Kff": Kff}

    def sample_next(xnew, outside_vars):
        """Repeatedly samples from the Gaussian process posterior,
        conditioning on previously sampled values.
        """


        # Variables from outer scope
        X, y, Kff = outside_vars["X"], outside_vars["y"], outside_vars["Kff"]

        # Compute Cholesky decomposition of kernel matrix
        Lff = Kff.cholesky()
        y_residual = y - model.mean_function(X)

        # Compute conditional mean and variance
        loc, cov = conditional(xnew, X, model.kernel, y_residual, None, Lff)
        if not noiseless:
            cov = cov + noise

        ynew = dist.Normal(loc + model.mean_function(xnew), cov.sqrt()).rsample()

        # Update kernel matrix
        N = outside_vars["N"]
        Kffnew = Kff.new_empty(N+1, N+1)
        Kffnew[:N, :N] = Kff
        cross = model.kernel(X, xnew).squeeze()
        end = model.kernel(xnew, xnew).squeeze()
        Kffnew[N, :N] = cross
        Kffnew[:N, N] = cross
        # No noise, just jitter for numerical stability
        Kffnew[N, N] = end + model.jitter
        # Heuristic to avoid adding degenerate points
        if Kffnew.logdet() > -15.:
            outside_vars["Kff"] = Kffnew
            outside_vars["N"] += 1
            outside_vars["X"] = torch.cat((X, xnew))
            outside_vars["y"] = torch.cat((y, ynew))

        return ynew

    return lambda xnew: sample_next(xnew, outside_vars)

def new_iter_sample(model, X, noiseless=True):

    with torch.no_grad():
        _mean, _cov = model(X[:,:-1], full_cov=True, noiseless=noiseless)

    n = _mean.size(0)
    jitter = torch.eye(n) * 1e-6

    L = torch.cholesky(_cov + jitter)
    V = torch.randn(X.size(0))

    mean = _mean + L @ V

    return mean

def smooth_posterior_sample(model, X_init, uniform_noise=False):

    X = X_init.clone().detach()
    V = torch.randn(X.size(0))

    with torch.no_grad():
        mean, cov = model(X[:,:-1], full_cov=True, noiseless=True)

    if not uniform_noise:
        noise = dist.Normal(0, cov.diag() + model.noise + model.jitter).sample()
    else:
        v = cov.diag().median()
        noise = dist.Normal(0, v + model.noise + model.jitter).sample(torch.tensor([X.size(0)]))

    n = mean.size(0)
    jitter = torch.eye(n) * model.jitter
    L = torch.cholesky(cov + jitter)
    sample = mean + L @ V

    outside_vars = {"X": X_init, "V": V, "noise": noise}

    def sample_next(xnew, outside_vars, return_full=False):
        """Repeatedly samples from the Gaussian process posterior,
        conditioning on previously sampled values.
        """

        sample_size = xnew.size(0)
        vnew = torch.randn(sample_size)

        # get variables from outer scope
        X, V, noise = outside_vars.get("X"), outside_vars.get("V"), outside_vars.get("noise")

        X = torch.cat((X, xnew))
        V = torch.cat((V, vnew))

        # ask for the noiseless covariance matrix
        # explicitly add noise to it when sampling new observation noise (separately)
        with torch.no_grad():
            mean, cov = model(X[:,:-1], full_cov=True, noiseless=True)

        if not uniform_noise:
            observation_variance = cov.diag()[-sample_size:]
            observation_noise = dist.Normal(0, observation_variance + model.noise + model.jitter).sample()
        else:
            print('sampling uniform observation noise')
            v = cov.diag().median()
            observation_noise = dist.Normal(0, v + model.noise + model.jitter).sample(torch.tensor([sample_size]))

        n = mean.size(0)
        jitter = torch.eye(n) * model.jitter
        L = torch.cholesky(cov + jitter)

        sample = mean + L @ V

        outside_vars["X"] = X
        outside_vars["V"] = V
        outside_vars["noise"] = torch.cat((outside_vars["noise"], observation_noise))

        if return_full:
            return sample, outside_vars["noise"]

        return sample[-sample_size:], observation_noise

    f = lambda x, return_full=False: sample_next(x, outside_vars, return_full=return_full)
    return f, sample, noise

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
    kernel = gp.kernels.RBF(input_dim=2, variance=torch.tensor(1.), lengthscale=torch.tensor([1.0, 1.0]))
    # kernel = gp.kernels.Matern52(input_dim=2, variance=torch.tensor(1.), lengthscale=torch.tensor([1.0, 1.0]))
    model = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(initial_noise_var), jitter=1e-8)
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

    def __init__(self, inputs, targets, optimize_noise_variance=True, dtype=DTYPE, num_steps=2000, lr=1e-3):
        """ fit independent GP models for each target
        inputs X and targets should be data frames
        """

        self.lr = lr
        self.dtype = dtype
        self.inputs = inputs
        self.targets = targets
        self.num_steps = num_steps
        self.optimize_noise_variance = optimize_noise_variance

        self.X = torch.tensor(inputs.values, dtype=dtype)
        self.Y = torch.tensor(targets.values, dtype=dtype)

        self.models = {}
        self.samplers = {}
        self.fit(num_steps=2000)

    def fit(self, lr=None, num_steps=None):
        if lr is None:
            lr = self.lr
        if num_steps is None:
            num_steps = self.num_steps

        for target in self.targets.keys():
            print(f'optimizing {target} model')
            y = torch.tensor(self.targets[target].values, dtype=self.dtype)

            model = model_ternary(self.X, y, initial_noise_var=0.01)
            update_posterior(model, num_steps=num_steps, lr=lr, optimize_noise_variance=self.optimize_noise_variance)
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

    def clean_iter_sample(self, X, target=None, noiseless=True, uniform_noise=False):
        """ sample GP posteriors iteratively"""

        if X.ndimension() == 1:
            X = X.unsqueeze(0)

        if target is None:
            targets = self.models.keys()
        elif type(target) is str:
            targets = [target]

        mean, noise = {}, {}
        for target in targets:
            try:
                fn = self.samplers[target]
                mean[target], noise[target] = fn(X)
            except KeyError:
                model = self.models[target]
                fn, mean[target], noise[target] = smooth_posterior_sample(model, X, uniform_noise=uniform_noise)
                self.samplers[target] = fn

        return mean, noise

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
