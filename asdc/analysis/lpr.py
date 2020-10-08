from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData, Status
from asdc._slack import SlackHandler

logger = logging.getLogger(__name__)

def current_crosses_zero(df: pd.DataFrame) -> bool:
    current= df['current']
    logger.debug('LPR check')
    return  current.min() < 0 and current.max() > 0

def _scan_range(df, potential_window=0.005) -> tuple[float, float]:
    current, potential = df['current'], df['potential']

    # find rough open circuit potential -- find zero crossing of current trace
    # if the LPR fit is any good, then the intercept should give
    # a more precise estimate of the open circuit potential
    zcross = np.argmin(np.abs(current))
    ocp = potential.iloc[zcross]

    # select a window around OCP to fit
    lb, ub = ocp - potential_window, ocp + potential_window
    return lb, ub

def valid_scan_range(df: EchemData, potential_window: float = 0.005) -> bool:
    current, potential = df['current'], df['potential']
    lb, ub = _scan_range(df, potential_window=potential_window)

    return potential.min() <= lb and potential.max() >= ub

def polarization_resistance(df: EchemData, potential_window: float = 0.005) -> tuple[float, float, float]:
    current, potential = df['current'].values, df['potential'].values

    lb, ub = _scan_range(df, potential_window=potential_window)
    fit_p = (potential >= lb) & (potential <= ub)

    # quick linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(current[fit_p], potential[fit_p])

    r2 = r_value**2

    return slope, intercept, r2

class LPRData(EchemData):

    @property
    def _constructor(self):
        return LPRData

    @property
    def name(self):
        return 'LPR'

    def check_quality(df, r2_thresh=0.9999, w=5):
        """ log results of quality checks and return a status code for control flow in the caller """

        status = Status.OK

        if not current_crosses_zero(df):
            logger.warning('LPR current does not cross zero!')
            status = max(status, Status.WARN)

        if not valid_scan_range(df, potential_window=w * 1e-3):
            logger.warning(f'scan range does not span +/- {w} mV')
            status = max(status, Status.WARN)

        slope, intercept, r2 = polarization_resistance(df)

        if r2 < r2_thresh:
            logger.warning('R^2 threshold not met')
            status = max(status, Status.WARN)

        logger.info(f'LPR slope: {slope} (R2={r2}), OCP: {intercept}')
        return status

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
