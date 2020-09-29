import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)

def current_crosses_zero(df):
    current = df['current']
    success = current.min() < 0 and current.max() > 0

    if not success:
        logger.warning('Tafel current does not cross zero!')

    logger.debug('Tafel check')

    return success

class TafelData(EchemData):

    @property
    def _constructor(self):
        return TafelData

    @property
    def name(self):
        return 'Tafel'

    def check_quality(self):
        return current_crosses_zero(self)

    def plot(self, fit=False):
        # # super().plot('current', 'potential')
        plt.plot(self['potential'], np.log10(np.abs(self['current'])))
        plt.axvline(0, color='k', alpha=0.5, linewidth=0.5)
        plt.xlabel('potential (V)')
        plt.ylabel('log current (A)')

        # if fit:
        #     ylim = plt.ylim()
        #     x = np.linspace(self.current.min(), self.current.max(), 100)
        #     slope, intercept, r_value = polarization_resistance(self)
        #     plt.plot(x, intercept + slope * x, linestyle='--', color='k', alpha=0.5)
        #     plt.ylim(ylim)

        plt.tight_layout()
