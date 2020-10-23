import lmfit
import numpy as np

from asdc.analysis.echem_data import EchemData

def butler_volmer(x, E_oc, j0, alpha_c, alpha_a):
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
    """ model current under butler-volmer model

    Example:
        ```
        bv = butler_volmer.ButlerVolmerModel()
        pars = bv.guess(tafel)
        E, logI = bv.slice(tafel, pars['E_oc'], w=0.1)
        bv_fit = bv.fit(logI, x=E, params=pars)
        ```
    """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(butler_volmer, **kwargs)

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

    def guess(self, data: EchemData, **kwargs):

        E = data.potential.values
        I = data.current.values

        # guess open circuit potential: minimum log current
        id_oc = np.argmin(np.abs(I))
        E_oc_guess = E[id_oc]

        E, I = self.slice(data, E_oc_guess)

        # guess corrosion current
        i_corr = np.max(I)

        pars = self.make_params(E_oc=E_oc_guess, j0=i_corr, alpha_c=0.5, alpha_a=0.5)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def slice(self, data: EchemData, E_oc: float, w: float = 0.15):
        E = data.potential.values
        I = data.current.values

        slc = (E > E_oc - w) & (E < E_oc + w)
        return E[slc], I[slc]


def butler_volmer_nuc(x, E_oc, i_corr, alpha_c, alpha_a, E_nuc, A, p, i_pass):
    overpotential = x - E_oc
    driving_force = x - E_nuc

    # nucleation model
    S = np.exp(-A * driving_force**p)

    # see eq 5 in Bellezze et al (10.1016/j.corsci.2017.10.012)
    current = i_corr * (S * np.exp(alpha_a * overpotential) - np.exp(-alpha_c * overpotential)) + (1-S)*i_pass
    return current


class ButlerVolmerNucleationModel(lmfit.Model):
    """ model current under butler-volmer model with a nucleation and growth active/passive effect

    Example:
        ```
        bv = butler_volmer.ButlerVolmerModel()
        pars = bv.guess(tafel)
        E, logI = bv.slice(tafel, pars['E_oc'], w=0.1)
        bv_fit = bv.fit(logI, x=E, params=pars)
        ```
    """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(butler_volmer_nuc, **kwargs)

    def _set_paramhints_prefix(self):
        """ E_oc, i_corr, alpha_c, alpha_a, E_nuc, A, p, i_pass """
        self.set_param_hint('i_corr', min=0)
        self.set_param_hint('alpha_c', min=0)
        self.set_param_hint('alpha_a', min=0)
        self.set_param_hint('A', min=1e-4, max=1e-3)
        self.set_param_hint('p', min=2, max=3)
        self.set_param_hint('i_pass', min=0)

    def guess(self, data: EchemData, **kwargs):
        """ E_oc, i_corr, alpha_c, alpha_a, E_nuc, A, p, i_pass """

        E = data.potential.values
        I = data.current.values

        # guess open circuit potential: minimum log current
        id_oc = np.argmin(np.abs(I))
        E_oc_guess = E[id_oc]

        E, I = self.slice(data, E_oc_guess)

        # guess corrosion current
        i_corr = np.max(I)

        pars = self.make_params(
            E_oc=E_oc_guess,
            i_corr=i_corr,
            alpha_c=0.5,
            alpha_a=0.5,
            E_nuc=E_oc_guess + 0.2,
            A=5e-4
        )
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def slice(self, data: EchemData, E_oc: float, w: float = 0.15):
        E = data.potential.values
        I = data.current.values

        slc = (E > E_oc - w) & (E < E_oc + w)
        return E[slc], I[slc]
