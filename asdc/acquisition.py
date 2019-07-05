import numpy as np

import torch
import torch.distributions as dist

alpha = {
    'I_p': 1,
    'slope': 1,
    'V_oc': 1,
    'V_tp': 1

}
# maximization problem:
# e.g. maximize the negative passivation current...
sign = {
    'I_p': -1,
    'slope': -1,
    'V_oc': -1,
    'V_tp': 1
}

def sample_weights(alpha=alpha):
    """ sample objective function weighting from Dirichlet distribution
    specified by concentration parameters alpha
    """
    a = torch.tensor(list(alpha.values()), dtype=torch.float)
    weights = dist.Dirichlet(a).sample()
    return dict(zip(alpha.keys(), weights))

def random_scalarization_cb(model, candidates, cb_beta, weights=None, sign=sign):
    # scaling some data by some constant --> scaling the variance by the square of the constant

    objective = torch.zeros(candidates.size(0))

    with torch.no_grad():
        mean, var = model(candidates)

    for key, weight in weights.items():
        m = model.models[key]
        mu, v = mean[key], var[key]

        # remap objective function to [0,1]
        # use the observed data to set the scaling.
        min_val = (m.y * sign[key]).min()
        scale = torch.abs(m.y.max() - m.y.min())

        mu = mu * sign[key]
        mu = (mu - min_val) / scale
        v = v * (1/scale)**2

        sd = v.sqrt()
        ucb = mu + np.sqrt(cb_beta) * sd
        objective += weight * ucb

    return objective
