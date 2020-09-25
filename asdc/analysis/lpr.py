import numpy as np
from scipy import stats

def current_crosses_zero(df):
    current = df['current']
    return current.min() < 0 and current.max() > 0

def polarization_resistance(df, current_window=2e-5):
    current, potential = df['current'].values, df['potential'].values

    # slice out a symmetric window around zero current
    window = (current > -current_window) & (current < current_window)
    x, y = current[window], potential[window]

    # quick linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return slope, r_value
