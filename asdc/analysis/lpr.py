import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData
from asdc._slack import SlackHandler

logger = logging.getLogger(__name__)

def current_crosses_zero(df):
    current = df['current']
    success = current.min() < 0 and current.max() > 0

    if not success:
        logger.warning('LPR current does not cross zero!')

    logger.debug('LPR check')

    return success

def polarization_resistance(df, current_window=2e-5):
    current, potential = df['current'].values, df['potential'].values

    # slice out a symmetric window around zero current
    window = (current > -current_window) & (current < current_window)
    x, y = current[window], potential[window]

    # quick linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return slope, intercept, r_value

def lpr_analysis(df, lim=.005, r2thresh=.97):
    x, y = df['current'].values, df['potential'].values

    # find rough open circuit potential
    zcross = np.argmin(np.abs(x))
    ocp = y[zcross]

    # make sure there is at least on point in the scan on either side of OCP
    flag1a = np.sum(y>ocp+lim) > 1
    flag1b = np.sum(y<ocp-lim) > 1
    flag1 = flag1a and flag1b

    # select `lim` mV window around OCP to fit
    fit_p = (y>ocp-lim)&(y<ocp+lim)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x[fit_p], y[fit_p])
    flag2 = r_value**2 > r2thresh

    flag = flag1 and flag2

    print(r_value**2)
    print(slope)

    output={'slope': slope,'intercept': intercept,'r2': r_value**2,'good_scan': flag,'ocp': ocp}

    return output

class LPRData(EchemData):

    @property
    def _constructor(self):
        return LPRData

    @property
    def name(self):
        return 'LPR'

    def check_quality(self):
        return current_crosses_zero(self)

    def plot(self, fit=False):
        # # super().plot('current', 'potential')
        plt.plot(self['current'], self['potential'])
        plt.axvline(0, color='k', alpha=0.5, linewidth=0.5)
        plt.xlabel('current (A)')
        plt.ylabel('potential (V)')

        if fit:
            ylim = plt.ylim()
            x = np.linspace(self.current.min(), self.current.max(), 100)
            slope, intercept, r_value = polarization_resistance(self)
            plt.plot(x, intercept + slope * x, linestyle='--', color='k', alpha=0.5)
            plt.ylim(ylim)

        plt.tight_layout()
