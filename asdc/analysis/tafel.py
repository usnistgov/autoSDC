import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData
from asdc.analysis import butler_volmer

import tafel_fitter.tafel as tafelfit

logger = logging.getLogger(__name__)


def current_crosses_zero(df):
    """ verify that a valid Tafel scan should has a current trace that crosses zero """
    current = df["current"]
    success = current.min() < 0 and current.max() > 0

    if not success:
        logger.warning("Tafel current does not cross zero!")

    logger.debug("Tafel check")

    return success


def fit_bv(df, w=0.2):
    bv = butler_volmer.ButlerVolmerLogModel()
    pars = bv.guess(df)
    E, logI = bv.slice(df, pars["E_oc"], w=w)

    # weights = np.square(1e-6 + np.abs(E - pars["E_oc"]))
    # weights = 1 / np.square(np.abs(logI))
    weights = 1 / np.exp(logI)
    # weights = np.square(np.exp(logI))

    for p in ("alpha_c", "alpha_a"):
        pars[p].set(vary=False)
    pars["i_corr"].set(max=10 ** (logI.max()))

    bv_fit = bv.fit(logI, x=E, params=pars, weights=weights, method="leastsq")

    refinement_pars = bv_fit.params
    # refinement_pars["i_corr"].set(max=10**(logI.max()))
    # refinement_pars["E_oc"].set(vary=False)
    for p in ("alpha_c", "alpha_a"):
        refinement_pars[p].set(vary=True)

    # weights = np.square(np.exp(logI))

    r = bv_fit.model.fit(
        logI, x=E, params=refinement_pars, weights=weights, method="lbfgsb"
    )

    return r


class TafelData(EchemData):
    @property
    def _constructor(self):
        return TafelData

    @property
    def name(self):
        return "Tafel"

    def check_quality(self):
        model = fit_bv(self)
        i_corr = model.best_values["i_corr"]
        ocp = model.best_values["E_oc"]
        print(f"i_corr: {i_corr}")

        logger.info(f"Tafel: OCP: {ocp}, i_corr: {i_corr}")
        return current_crosses_zero(self)

    def fit_bv(self, w=0.2):
        """ fit a butler volmer model to Tafel data """
        self.model = fit_bv(self, w=w)

        # convenience attributes:
        # just store optimized model params in class attributes for now
        self.i_corr = self.model.best_values["i_corr"]
        self.ocp = self.model.best_values["E_oc"]
        self.alpha_c = self.model.best_values["alpha_c"]
        self.alpha_a = self.model.best_values["alpha_a"]

        return self.model

    def fit(self):
        self.ocp = tafelfit.estimate_ocp(
            self["potential"].values, self["current"].values
        )

        u = self["potential"].values - self.ocp
        tafel_data, fits = tafelfit.tafel_fit(
            u, self["current"].values, windows=np.arange(0.025, 0.25, 0.001)
        )

        self.tafel_data = tafel_data
        self.i_corr = (tafel_data["cathodic"]["j0"] + tafel_data["anodic"]["j0"]) / 2

        self.alpha_c = tafel_data["cathodic"]["dlog(j)/dV"]
        self.alpha_a = tafel_data["anodic"]["dlog(j)/dV"]

    def evaluate_model(self, V_mod=None):
        """ evaluate butler-volmer model on regular grid """
        if V_mod is None:
            V_mod = np.linspace(
                self.potential.min() - 0.5, self.potential.max() + 0.5, 200
            )

        I_mod = self.model.eval(self.model.params, x=V_mod)
        return V_mod, I_mod

    def plot(self, fit=False, w=0.2):
        """ Tafel plot: log current against the potential """
        # # super().plot('current', 'potential')
        plt.plot(self["potential"], np.log10(np.abs(self["current"])))
        plt.xlabel("potential (V)")
        plt.ylabel("log current (A)")
        ylim = plt.ylim()
        xlim = plt.xlim()

        if fit:
            ylim = plt.ylim()
            self.fit()
            overpotential = self["potential"] - self.ocp

            lims = plt.gca().get_ylim()

            colors = ["g", "m"]
            for idx, (segment, best_fit) in enumerate(self.tafel_data.items()):

                plt.plot(
                    self["potential"].values,
                    np.log10(best_fit["j0"]) + best_fit["dlog(j)/dV"] * overpotential,
                    color=colors[idx],
                )
                plt.axhline(
                    np.log10(best_fit["j0"]), label=f"j0 {segment}", color=colors[idx]
                )
                plt.axvline(
                    self.ocp + best_fit["window_min"],
                    color="k",
                    alpha=0.2,
                    linestyle="--",
                )
                plt.axvline(
                    self.ocp + best_fit["window_max"],
                    color="k",
                    alpha=0.2,
                    linestyle="--",
                )

            plt.ylim(*lims)

            log_i_corr = np.log10(self.i_corr)
            plt.axhline(log_i_corr, color="k", alpha=0.5, linewidth=0.5)

            plt.ylim(ylim)
            plt.xlim(xlim)

        plt.tight_layout()

    def plot_bv(self, fit=False, w=0.2):
        """ Tafel plot: log current against the potential """
        # # super().plot('current', 'potential')
        plt.plot(self["potential"], np.log10(np.abs(self["current"])))
        plt.xlabel("potential (V)")
        plt.ylabel("log current (A)")
        ylim = plt.ylim()
        xlim = plt.xlim()

        if fit:
            ylim = plt.ylim()
            model = self.fit(w=w)

            # evaluate and plot model
            V, I_mod = self.evaluate_model()

            overpotential = V - self.ocp
            bc = self.alpha_c / np.log(10)
            ba = self.alpha_a / np.log(10)
            log_i_corr = np.log10(self.i_corr)

            plt.plot(V, I_mod, linestyle="--", color="k", alpha=0.5)
            plt.axhline(log_i_corr, color="k", alpha=0.5, linewidth=0.5)

            cpt_style = dict(color="k", alpha=0.5, linewidth=0.5)

            # cathodic branch
            plt.plot(V, -overpotential * bc + log_i_corr, **cpt_style)

            # anodic branch
            plt.plot(V, overpotential * ba + log_i_corr, **cpt_style)

            plt.ylim(ylim)
            plt.xlim(xlim)

        plt.tight_layout()
