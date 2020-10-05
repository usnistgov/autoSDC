import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData
from asdc._slack import SlackHandler

logger = logging.getLogger(__name__)

def current_crosses_zero(current):
    success = current.min() < 0 and current.max() > 0

    if not success:
        logger.warning('LPR current does not cross zero!')

    logger.debug('LPR check')

    return success

def polarization_resistance(current, potential, potential_window=0.005):

    # find rough open circuit potential -- find zero crossing of current trace
    # if the LPR fit is any good, then the intercept should give
    # a more precise estimate of the open circuit potential
    zcross = np.argmin(np.abs(current))
    ocp = potential[zcross]

    # select a window around OCP to fit
    lb, ub = ocp - potential_window, ocp + potential_window
    fit_p = (potential >= lb) & (potential <= ub)

    # quick linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(current[fit_p], potential[fit_p])

    r2 = r_value**2

    return slope, intercept, r2

def lpr_analysis(df, potential_window=.005, r2thresh=.97):
    current, potential = df['current'].values, df['potential'].values

    # find rough open circuit potential -- find zero crossing of current trace
    # if the LPR fit is any good, then the intercept should give
    # a more precise estimate of the open circuit potential
    zcross = np.argmin(np.abs(current))
    ocp = potential[zcross]

    # make sure there is at least on point in the scan on either side of OCP
    flag1a = np.sum(potential > ocp + potential_window) > 1
    flag1b = np.sum(potential < ocp - potential_window) > 1
    flag1 = flag1a and flag1b

    # select `lim` mV window around OCP to fit
    fit_p = (potential > ocp-potential_window) & (potential < ocp+potential_window)

    slope, intercept, r_value, p_value, std_err = stats.linregress(current[fit_p], potential[fit_p])
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
        slope, intercept, r2 = polarization_resistance(self.current, self.potential)
        print(f'OCP: {intercept}')
        return current_crosses_zero(self.current)

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
