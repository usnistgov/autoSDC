import gpflow
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

def model_ternary(composition, target):

    X = composition[:,:-1] # ignore the last composition column
    Y = target

    # sel = np.isfinite(Y).sum(axis=1)
    sel = np.isfinite(Y).flat
    X, Y = X[sel], Y[sel]
    N, D = X.shape

    gpflow.reset_default_graph_and_session()

    with gpflow.defer_build():
        m = gpflow.models.GPR(
            X, Y,
            # kern=gpflow.kernels.Linear(D, ARD=True) + gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D)
            kern=gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D, variance=0.01)
        )

    # set a weakly-informative lengthscale prior
    # e.g. half-normal(0, dx/3) -> gamma(0.5, 2*dx/3)
    # another choice might be to use an inverse gamma prior...
    m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(0.5, 2.0/3)
    m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    m.kern.kernels[1].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    m.kern.kernels[2].variance.prior = gpflow.priors.Gamma(0.5, 2.)

    m.compile()
    return m

class ExperimentEmulator():
    def __init__(self, datafile, components=['Ni', 'Al', 'Ti'], targets = ['V_oc', 'I_p', 'V_tp', 'slope', 'fwhm']):
        """ fit independent GP models for each target -- read compositions and targets from a csv file... """
        self.df = pd.read_csv(datafile, index_col=0)

        # drop the anomalous point 44 that has a negative jog in the passivation...
        self.df = self.df.drop(44)

        self.components = components
        self.targets = targets

        for component in self.components:
            self.df[component] = self.df[component] / 100

        self.models = {}
        self.fit()

    def fit(self):
        self.composition = self.df.loc[:,self.components].values

        self.opt = gpflow.training.ScipyOptimizer()
        for target in self.targets:
            model = model_ternary(self.composition, self.df[target].values[:,None])
            session = gpflow.get_default_session()
            self.opt.minimize(model)
            self.models[target] = (session, model)

    def __call__(self, composition, target=None, return_var=False):
        """ evaluate GP models on compositions """
        session, model = self.models[target]

        with session.as_default():
            mu, var = model.predict_y(composition[:,:-1])
            if return_var:
                return mu, var
            else:
                return mu.flatten()
