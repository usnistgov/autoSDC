import matplotlib.pyplot as plt

from asdc import visualization

def plot_emulator(model, domain, sample_posterior=True):
    if sample_posterior:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,12))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

    features = ['V_oc', 'V_tp', 'I_p', 'slope']
    labels = [r'$V_{oc}$', r'$V_{tp}$', r'log $I_p$', 'slope']

    mean, var = model(domain)
    if sample_posterior:
        # sample = model(domain, sample_posterior=True, noiseless=False)
        model.samplers = {}
        Y_sample, noise_sample = model.clean_iter_sample(domain, uniform_noise=True)
        sample = {key: (Y_sample[key] + noise_sample[key]) for key in Y_sample.keys()}

    for col, feature, title in zip(axes.T, features, labels):
        m = model.models[feature]
        visualization.ternary_scatter_sub(m.X, m.y, ax=col[0])
        visualization.ternary_scatter_sub(domain.numpy(), mean[feature].numpy(), ax=col[1]);
        if sample_posterior:
            visualization.ternary_scatter_sub(domain.numpy(), sample[feature].numpy(), ax=col[2]);

        for ax in col:
            ax.axis('equal')
        col[0].set_title(title, size=18)

    plt.tight_layout()
    plt.subplots_adjust()
