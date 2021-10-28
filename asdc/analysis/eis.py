import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class PotentiostaticEISData(EchemData):
    @property
    def _constructor(self):
        return PotentiostaticEISData

    @property
    def name(self):
        return "PotentiostaticEIS"

    def check_quality(self):
        """ Potentiostatic EIS heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot Potentiostatic: current vs potential """
        plt.plot(self["frequency"], self["impedance_real"])
        plt.xlabel("frequency (Hz)")
        plt.ylabel("Re(impedance)")
        plt.semilogx()

        plt.tight_layout()
