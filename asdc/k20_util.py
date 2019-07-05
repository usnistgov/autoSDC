import matplotlib.pyplot as plt

from asdc import visualization

def plot_emulator(model, domain, sample_posterior=True, fig_path=None):
    n_objectives = len(model.models)

    if sample_posterior:
        fig, axes = plt.subplots(nrows=3, ncols=n_objectives, figsize=(4*n_objectives,12))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=n_objectives, figsize=(4*n_objectives,8))

    labels = {
        'V_oc': r'$V_{oc}$',
        'V_tp': r'$V_{tp}$',
        'I_p': r'log $I_p$',
        'slope': 'slope'
    }

    mean, var = model(domain)
    if sample_posterior:
        # sample = model(domain, sample_posterior=True, noiseless=False)
        model.samplers = {}
        Y_sample, noise_sample = model.clean_iter_sample(domain, uniform_noise=True)
        sample = {key: (Y_sample[key] + noise_sample[key]) for key in Y_sample.keys()}

    for col, feature, m in zip(axes.T, model.models.items()):

        visualization.ternary_scatter_sub(m.X, m.y, ax=col[0])
        visualization.ternary_scatter_sub(domain.numpy(), mean[feature].numpy(), ax=col[1]);

        if sample_posterior:
            visualization.ternary_scatter_sub(domain.numpy(), sample[feature].numpy(), ax=col[2]);

        for ax in col:
            ax.axis('equal')
        col[0].set_title(labels[feature], size=18)

    plt.tight_layout()
    plt.subplots_adjust()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()
