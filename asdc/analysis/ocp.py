import csaps
import numpy as np

def ocp_stop(x, y, time=90, tstart=300, thresh=.00003):
    t = tstart
    deriv = np.inf
    while (np.abs(deriv)>thresh) and (t<900):
        tstop = t + time
        deriv = np.mean(y[(x > t) & (x < tstop)])
        t = tstop
    return t

def ocp_check(ocp, smooth=0.001, tr=100):
    """ model an open circuit potential trace to check that it converges to a constant value

    computes the average slope at the end of the potential trace
    the RMS error of a cubic spline model
    the maximum potential jump over a single measurement interval
    and the hold stop time
    """
    t, potential = ocp['elapsed_time'], ocp['potential']

    # estimate the derivative using a cubic spline model
    model = csaps(t, potential, smooth=smooth)
    dVdt = model.spline.derivative()(t)

    tstop = ocp_stop(t, dVdt)

    # average the smoothed derivative over the last time chunk
    checktime = t.max() - tr
    avslope = np.mean(dVdt[t > checktime])

    # compute the largest spike in the finite difference derivative
    maxdiff = np.max(np.abs(np.diff(potential)))

    # compute the root-mean-square error of the spline model
    rms = np.sqrt(np.mean((model.spline(t) - potential)**2))

    results = {
        'average_slope': avslope,
        'rms': rms,
        'spike': maxdiff,
        'stop time': tstop
    }

    return results
