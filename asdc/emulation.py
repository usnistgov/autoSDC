import gpflow
import dataset
import numpy as np
import pandas as pd

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

def model_ternary(composition, target, reset_tf_graph=True, drop_last=True, optimize_noise_variance=True, initial_noise_var=1e-4):

    if drop_last:
        X = composition[:,:-1] # ignore the last composition column
    else:
        X = composition
    Y = target

    # sel = np.isfinite(Y).sum(axis=1)
    sel = np.isfinite(Y).flat
    X, Y = X[sel], Y[sel]
    N, D = X.shape

    if reset_tf_graph:
        gpflow.reset_default_graph_and_session()

    with gpflow.defer_build():
        m = gpflow.models.GPR(
            X, Y,
            # kern=gpflow.kernels.Linear(D, ARD=True) + gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D)
            kern=gpflow.kernels.Matern52(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D, variance=initial_noise_var) # \sigma_noise = 0.01
            # kern=gpflow.kernels.RationalQuadratic(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D, variance=initial_noise_var)
        )

    # set a weakly-informative lengthscale prior
    # e.g. half-normal(0, dx/3) -> gamma(0.5, 2*dx/3)
    # another choice might be to use an inverse gamma prior...
    # m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(0.5, 2.0/3)
    m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(0.5, 0.5)

    # m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    # m.kern.kernels[1].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    # m.kern.kernels[2].variance.prior = gpflow.priors.Gamma(0.5, 2.)
    m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2.0, 2.0)
    m.kern.kernels[1].variance.prior = gpflow.priors.Gamma(2.0, 2.0)
    # m.kern.kernels[2].variance.prior = gpflow.priors.Gamma(2.0, 2.0)

    if not optimize_noise_variance:
        m.kern.kernels[2].variance.trainable = False

    m.likelihood.variance = 1e-6

    m.compile()
    return m

def model_synth(X, y, dx=1.0):
    D = X.shape[1]

    with gpflow.defer_build():
        model = gpflow.models.GPR(
            X, y,
            kern=gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D),
        )

        model.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2,1/2)
        model.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(2.0, 2*dx/3)
        model.likelihood.variance = 0.01

    model.compile()
    return model

def model_bounded(X, y, dx=1.0):
    D = X.shape[1]
    with gpflow.defer_build():
        model = gpflow.models.VGP(
            X, y,
            kern=gpflow.kernels.RBF(D, ARD=True),
            likelihood=gpflow.likelihoods.Beta()
        )

        model.kern.variance.prior = gpflow.priors.Gamma(2,2)
        model.kern.lengthscales.prior = gpflow.priors.Gamma(1.0, 2*dx/3)
        model.likelihood.variance = 0.1

    model.compile()
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
