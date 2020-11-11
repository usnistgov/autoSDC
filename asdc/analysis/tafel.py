import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData
from asdc.analysis import butler_volmer

logger = logging.getLogger(__name__)

def current_crosses_zero(df):
    """ verify that a valid Tafel scan should has a current trace that crosses zero """
    current = df['current']
    success = current.min() < 0 and current.max() > 0

    if not success:
        logger.warning('Tafel current does not cross zero!')

    logger.debug('Tafel check')

    return success

def fit_bv(df, w=0.2):
    bv = butler_volmer.ButlerVolmerLogModel()
    pars = bv.guess(df)
    E, I = bv.slice(df, pars['E_oc'], w=w)
    bv_fit = bv.fit(I, x=E, params=pars)
    return bv_fit

class TafelData(EchemData):

    @property
    def _constructor(self):
        return TafelData

    @property
    def name(self):
        return 'Tafel'

    def check_quality(self):
        model = fit_bv(self)
        i_corr = model.best_values["j0"]
        ocp = model.best_values["E_oc"]
        print(f'i_corr: {i_corr}')

        logger.info(f'Tafel: OCP: {ocp}, i_corr: {i_corr}')
        return current_crosses_zero(self)

    def fit(self):
        model = fit_bv(self)
        i_corr = model.best_values["j0"]
        ocp = model.best_values["E_oc"]
        return ocp, i_corr

    def plot(self, fit=False):
        """ Tafel plot: log current against the potential """
        # # super().plot('current', 'potential')
        plt.plot(self['potential'], np.log10(np.abs(self['current'])))
        plt.xlabel('potential (V)')
        plt.ylabel('log current (A)')

        if fit:
            ylim = plt.ylim()
            model = fit_bv(self, w=0.2)
            # print(f'i_corr: {model.best_values["j0"]}')

            x = np.linspace(self.potential.min()-0.5, self.potential.max()+0.5, 200)
            I_mod = model.eval(model.params, x=x)
            plt.plot(x, I_mod, linestyle='--', color='k', alpha=0.5)
            plt.axhline(np.log10(model.best_values['j0']), color='k', alpha=0.5, linewidth=0.5)

            # nu = self.potential.values - model.best_values['E_oc']
            nu = x - model.best_values['E_oc']
            icorr = np.log10(model.best_values['j0'])
            bc = model.best_values['alpha_c'] / np.log(10)
            ba = model.best_values['alpha_a'] / np.log(10)

            # plt.plot(self.potential.values, -nu*bc + icorr, color='k', alpha=0.5, linewidth=0.5)
            # plt.plot(self.potential.values, nu*ba + icorr, color='k', alpha=0.5, linewidth=0.5)
            plt.plot(x, -nu*bc + icorr, color='k', alpha=0.5, linewidth=0.5)
            plt.plot(x, nu*ba + icorr, color='k', alpha=0.5, linewidth=0.5)


            plt.ylim(ylim)

        plt.tight_layout()
