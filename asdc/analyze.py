import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal

import lmfit
from lmfit import models

def model_autorange_artifacts(V, I, threshold=0.5, tau_increasing=10, tau_decreasing=3.8):
    """ autorange artifacts occur when the potentiostat switches current ranges
    The effect is an apparent spike in measured current by about an order of magnitude...
    This function attempts to detect and model these artifacts as step functions with exponential decay
    """
    artifact_model = np.zeros_like(I)

    # detect artifacts by thresholding the numerical derivative of
    # the log absolute current...
    dI = np.diff(np.log10(np.abs(I)))
    autorange_idx, = np.where(np.abs(dI) > threshold)

    # get the signed step height on the log current
    steps = dI[autorange_idx]

    # model each artifact as an exponentially decaying step function...
    for idx, step_magnitude in zip(autorange_idx, steps):

        # different relaxation times depending on current direction (and voltage ramp direction...)
        if step_magnitude > 0:
            tau = tau_increasing
        elif step_magnitude < 0:
            tau = tau_decreasing

        # offset the index of the step by 1 (due numpy.diff using a right-handed difference)
        pulse =  signal.exponential(artifact_model.size, center=idx+1, tau=tau, sym=False)

        # signal.exponential generates a symmetric window... zero out the left half
        pulse[:idx+1] = 0

        artifact_model += step_magnitude*pulse

    return artifact_model

def guess_open_circuit_potential(V, log_I):
    open_circuit = V[np.argmin(log_I)]
    return open_circuit

def model_open_circuit_potential(V, log_I):
    """ extract open circuit potential by modeling log(I)-V curve an laplace peak with a polynomial background """
    y = -log_I

    def laplace(x, loc, scale, amplitude):
        return amplitude * stats.laplace.pdf(x, loc=loc, scale=scale)

    peak = lmfit.Model(laplace, prefix='peak_')
    bg = models.PolynomialModel(3, prefix='bg_')

    loc = np.argmax(y)
    pars = bg.guess(y, x=V)
    pars += peak.make_params(peak_loc=V[loc], peak_scale=0.01, peak_amplitude=0.1)

    model = peak + bg
    fitted_model = model.fit(y, x=V, params=pars, nan_policy='omit')

    return fitted_model

def voltage_turning_points(V):
    dV = np.diff(signal.savgol_filter(V, 11, 4))

    # find zero-crossings on the derivative...
    turning_points, = np.where(np.diff(np.sign(dV)))
    return turning_points

def segment_IV(I, V, segment=1):
    """ by default give the middle segment """
    t = voltage_turning_points(V)

    if segment is not None:
        zeros = voltage_turning_points(V)
        if segment == 0:
            I = I[:t[0]]
            V = V[:t[0]]

        elif segment == 1:
            I = I[t[0]:t[1]]
            V = V[t[0]:t[1]]
        elif segment == 2:
            I = I[t[1]:]
            V = V[t[1]:]

    return I, V
