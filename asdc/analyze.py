import numpy as np
import pandas as pd
from scipy import signal

def model_autorange_defects(V, I, threshold=0.5):
    """ autorange defects occur when the potentiostat switches current ranges
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
            tau = 10
        elif step_magnitude < 0:
            tau = 3.8

        # offset the index of the step by 1 (due numpy.diff using a right-handed difference)
        pulse =  signal.exponential(artifact_model.size, center=idx+1, tau=tau, sym=False)

        # signal.exponential generates a symmetric window... zero out the left half
        pulse[:idx+1] = 0

        artifact_model += step_magnitude*pulse

    return artifact_model


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
