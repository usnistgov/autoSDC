import lmfit
import numpy as np

from asdc.analysis.echem_data import EchemData

def butler_volmer(x, E_oc, j0, alpha_c, alpha_a):
    # alpha_a = 1 - alpha_c
    # alpha_a = alpha_c
    overpotential = x - E_oc
    current = j0 * (np.exp(alpha_a * overpotential) - np.exp(-alpha_c * overpotential))
    return current

def log_butler_volmer(x, E_oc, j0, alpha_c, alpha_a):
    abscurrent = np.abs(butler_volmer(x, E_oc, j0, alpha_c, alpha_a))

    # clip absolute current values so that the lmfit model
    # does not produce NaN values when evaluating the log current
    # at the exact open circuit potential
    return np.log10(np.clip(abscurrent, 1e-9, np.inf))

class ButlerVolmerModel(lmfit.Model):
    """ model log current under butler-volmer model """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(log_butler_volmer, **kwargs)

    def _set_paramhints_prefix(self):
        self.set_param_hint('j0', min=0)
        self.set_param_hint('alpha_c', min=0)
        self.set_param_hint('alpha_a', min=0)

    def _guess(self, data, x=None, **kwargs):
        # guess open circuit potential: minimum log current
        id_oc = np.argmin(data)
        E_oc_guess = x[id_oc]

        # unlog the data to guess corrosion current
        i_corr = np.max(10**data)

        pars = self.make_params(E_oc=E_oc_guess, j0=i_corr, alpha_c=0.5, alpha_a=0.5)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def guess(self, data, **kwargs):

        E = data.potential.values
        logI = np.log10(np.abs(data.current.values))

        # guess open circuit potential: minimum log current
        id_oc = np.argmin(logI)
        E_oc_guess = E[id_oc]

        E, logI = self.slice(data, E_oc_guess)

        # unlog the data to guess corrosion current
        i_corr = np.max(10**logI)

        pars = self.make_params(E_oc=E_oc_guess, j0=i_corr, alpha_c=0.5, alpha_a=0.5)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def slice(self, data, E_oc, w=0.15):
        E = data.potential.values
        logI = np.log10(np.abs(data.current.values))

        slc = (E > E_oc - w) & (E < E_oc + w)
        return E[slc], logI[slc]
