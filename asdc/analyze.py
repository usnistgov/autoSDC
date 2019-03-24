import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
from sklearn import linear_model
from skimage import filters

import lmfit
from lmfit import models

def laplace(x, loc, scale, amplitude):
    """ laplace peak shape for fitting open circuit potential on polarization curve  """
    return amplitude * stats.laplace.pdf(x, loc=loc, scale=scale)

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
        pulse = signal.exponential(artifact_model.size, center=idx+1, tau=tau, sym=False)

        # signal.exponential generates a symmetric window... zero out the left half
        pulse[:idx+1] = 0

        artifact_model += step_magnitude*pulse

    return artifact_model

def guess_open_circuit_potential(V, log_I):
    open_circuit = V[np.argmin(log_I)]
    return open_circuit

def model_open_circuit_potential(V, log_I, bg_order=3):
    """ extract open circuit potential by modeling log(I)-V curve an laplace peak with a polynomial background """
    y = -log_I

    peak = lmfit.Model(laplace, prefix='peak_')
    bg = models.PolynomialModel(bg_order, prefix='bg_')

    loc = np.argmax(y)
    pars = bg.guess(y, x=V)
    pars += peak.make_params(peak_loc=V[loc], peak_scale=0.01, peak_amplitude=0.1)

    model = peak + bg
    fitted_model = model.fit(y, x=V, params=pars, nan_policy='omit')

    return fitted_model

def to_odd(x):
    if x % 2:
        return x
    return x - 1

def piecewise_savgol(x, y, x_split=0, window_length=121, polyorder=5):
    """ piecewise smoothing with a savgol filter
    smooth y piecewise, splitting on x
    """
    sel = x <= x_split

    if sel.sum() < window_length:
        wl = to_odd(sel.sum() - 1)
    else:
        wl = window_length

    y[sel]  = signal.savgol_filter(y[sel], wl, polyorder)

    sel = x > x_split
    y[sel]  = signal.savgol_filter(y[sel], window_length, polyorder)

    return y

def model_polarization_curve(V, log_I, bg_order=3, smooth=True, shoulder_percentile=0.99, lm_method='huber'):
    """ extract features from polarization curve.
    open circuit potential by modeling log(I)-V curve an laplace peak with a polynomial background
    extract passivation region by fitting a robust regression model (with the laplace peak as a hint for where to start)
    """
    peak = model_open_circuit_potential(V, log_I, bg_order=bg_order)
    V_oc = peak.best_values['peak_loc']

    # apply piecewise smoothing
    # i.e. don't oversmooth the open circuit peak
    if smooth:
        log_I = piecewise_savgol(V, log_I, x_split=V_oc)

    peak_shoulder_idx = np.argmax(
        stats.laplace.cdf(V, loc=V_oc, scale=peak.best_values['peak_scale']) > shoulder_percentile
    )

    # fit robust regression model to passivation region
    vp = V[peak_shoulder_idx:]
    ip = log_I[peak_shoulder_idx:]
    if lm_method == 'thiel-sen':
        lm = linear_model.TheilSenRegressor()
    elif lm_method == 'huber':
        lm = linear_model.HuberRegressor()
    elif lm_method == 'ransac':
        lm = linear_model.RANSACRegressor()

    score = []
    n_fit = np.arange(50, 500, 5)
    for n in n_fit:

        lm.fit(vp[:n, None], ip[:n])
        score.append(lm.score(vp[:,None], ip))

    # refit with the best model...
    n = n_fit[np.argmin(score)]
    lm.fit(vp[:n, None], ip[:n])

    deviation = ip - lm.predict(vp[:,None])
    thresh = filters.threshold_triangle(deviation)
    id_thresh = np.argmax(deviation > thresh)
    V_transpassive = vp[id_thresh]

    I_passive = np.median(ip[:id_thresh])

    polarization_data = {
        'V_oc': V_oc,
        'V_tp': V_transpassive,
        'I_p': I_passive
    }

    return log_I, peak, lm, polarization_data

def extract_open_circuit_potential(current, potential, segment, return_model=False):

    # use the first CV cycle...
    sel = np.array(segment) == 2
    I = np.array(current)[sel]
    V = np.array(potential)[sel]

    # hack: use just the increasing ramp...
    # this works for 75 mV/s scan from -1V to 1.2V...
    I = I[:1000]
    V = V[:1000]

    # now correct for autorange artifacts
    a = model_autorange_artifacts(V, I, tau_increasing=10)
    model = model_open_circuit_potential(V, np.log10(np.abs(I)) - a)

    if return_model:
        return model

    return model.best_values['peak_loc']

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
