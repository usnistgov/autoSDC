import gpflow
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float

optimizer = gpflow.optimizers.Scipy()

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

def model_ternary(composition, target, drop_last=False, optimize_noise_variance=True, initial_noise_var=1e-4):

    if drop_last:
        X = composition[:,:-1] # ignore the last composition column
    else:
        X = composition
    Y = target

    # sel = np.isfinite(Y).sum(axis=1)
    sel = np.isfinite(Y).flat
    X, Y = X[sel], Y[sel]
    N, D = X.shape

    m = gpflow.models.GPR(
        data=(X,Y),
        # kernel=gpflow.kernels.Linear(D, ARD=True) + gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D),
        kernel=gpflow.kernels.Matern52(D, ARD=True) + gpflow.kernels.Constant(D),
        # kernel=gpflow.kernels.RationalQuadratic(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D, variance=initial_noise_var),
        noise_variance = 1e-3
    )

    # set a weakly-informative lengthscale prior
    # e.g. half-normal(0, dx/3) -> gamma(0.5, 2*dx/3)
    # another choice might be to use an inverse gamma prior...
    # m.kernel.kernels[0].lengthscales.prior = tpf.Gamma(f64(0.5), f64(2.0/3))
    m.kernel.kernels[0].lengthscales.prior = tfp.Gamma(f64(0.5), f64(0.5))

    # m.kernel.kernels[0].variance.prior = tpf.Gamma(f64(0.5), f64(4.))
    # m.kernel.kernels[1].variance.prior = tpf.Gamma(f64(0.5), f64(4.))
    m.kernel.kernels[0].variance.prior = tpf.Gamma(f64(2.0), f64(2.0))
    m.kernel.kernels[1].variance.prior = tpf.Gamma(f64(2.0), f64(2.0))

    if not optimize_noise_variance:
        gpflow.set_trainable(m.likelihood.variance, False)

    return m

def model_property(X, y, dx=1.0):

    sel = np.isfinite(y).flat
    X, y = X[sel], y[sel]
    N, D = X.shape

    ls = 10 * np.ones(D)

    model = gpflow.models.GPR(
        data=(X,y),
        # kernel=gpflow.kernels.RBF(lengthscales=ls),
        kernel=gpflow.kernels.RationalQuadratic(lengthscales=ls),
        mean_function=gpflow.mean_functions.Constant(np.median(y)),
        noise_variance=0.01
    )

    model.kernel.variance.prior = tfd.Gamma(f64(2), f64(1/2))
    model.kernel.lengthscales.prior = tfd.Gamma(f64(2.0), f64(2*dx/3))
    model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

    if optimize:
        optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options=dict(disp=True, maxiter=1000)
        )

    return model

def model_quality(X, y, dx=1.0, likelihood='beta', optimize=False):

    sel = np.isfinite(y).flat
    X, y = X[sel], y[sel]
    N, D = X.shape

    if likelihood == 'beta':
        # bounded regression
        lik = gpflow.likelihoods.Beta()
    elif likelihood == 'bernoulli':
        # classification
        lik = gpflow.likelihoods.Bernoulli()

    ls = np.ones(D)

    model = gpflow.models.VGP(
        data=(X, y),
        kern=gpflow.kernels.RBF(lengthscales=ls),
        likelihood=lik
    )

    model.kernel.variance.prior = tfd.Gamma(f64(2), f64(1/2))
    model.kernel.lengthscales.prior = tfd.Gamma(f64(2.0), f64(2*dx/3))


    if optimize:
        optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options=dict(disp=True, maxiter=1000)
        )

    return model

class NiTiAlEmulator():
    def __init__(self, composition, df, components=['Ni', 'Al', 'Ti'], targets = ['V_oc', 'I_p', 'V_tp', 'slope', 'fwhm'], dx=1.0):
        """ fit independent GP models for each target -- read compositions and targets from a csv file... """

        self.composition = composition
        self.components = list(composition.columns)
        self.df = df
        self.targets = targets
        # self.composition = self.df.loc[:,self.components].values
        self.dx = dx

        self.models = {}
        self.session = gpflow.get_default_session()
        self.opt = gpflow.training.ScipyOptimizer()

        self.fit()

    def fit(self):
        print(self.composition.shape)
        with self.session.as_default():
            for target in self.targets:

                model = model_property(self.composition.values[:,:-1], self.df[target].values[:,None], dx=self.dx)
                self.opt.minimize(model)
                self.models[target] = model

    def likelihood_variance(self, target=None):
        return self.models[target].likelihood.variance.value.item()

    def __call__(self, composition, target=None, return_var=False, sample_posterior=False, n_samples=1, seed=None):
        """ evaluate GP models on compositions """
        model = self.models[target]

        with self.session.as_default():
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
        self.targets = targets
        self.optimize_noise_variance = optimize_noise_variance

        self.models = {}
        self.fit()

    def fit(self):
        self.composition = self.df.loc[:,self.components].values

        self.opt = gpflow.training.ScipyOptimizer()
        for target in self.targets:
            model = model_ternary(self.composition, self.df[target].values[:,None], optimize_noise_variance=self.optimize_noise_variance)
            session = gpflow.get_default_session()
            self.opt.minimize(model)
            self.models[target] = (session, model)

    def __call__(self, composition, target=None, return_var=False, sample_posterior=False, n_samples=1, seed=None):
        """ evaluate GP models on compositions """
        session, model = self.models[target]

        with session.as_default():
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
