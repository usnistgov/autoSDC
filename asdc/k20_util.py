import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpflowopt.pareto import non_dominated_sort

from asdc import visualization

labels = {
    'V_oc': r'$V_{oc}$',
    'V_tp': r'$V_{tp}$',
    'I_p': r'log $I_p$',
    'slope': 'slope'
}

def plot_emulator(model, domain, sample_posterior=True, fig_path=None):
    n_objectives = len(model.models)

    if sample_posterior:
        fig, axes = plt.subplots(nrows=3, ncols=n_objectives, figsize=(4*n_objectives,12))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=n_objectives, figsize=(4*n_objectives,8))

    mean, var = model(domain)
    if sample_posterior:
        # sample = model(domain, sample_posterior=True, noiseless=False)
        model.samplers = {}
        Y_sample, noise_sample = model.clean_iter_sample(domain, uniform_noise=True)
        sample = {key: (Y_sample[key] + noise_sample[key]) for key in Y_sample.keys()}

    for col, (feature, m) in zip(axes.T, model.models.items()):

        visualization.ternary_scatter_sub(m.X, m.y, ax=col[0])
        visualization.ternary_scatter_sub(domain.numpy(), mean[feature].numpy(), ax=col[1], s=20, edgecolors=None, marker='H');

        if sample_posterior:
            visualization.ternary_scatter_sub(domain.numpy(), sample[feature].numpy(), ax=col[2], s=20, edgecolors=None, marker='H');

        for ax in col:
            ax.axis('equal')
        col[0].set_title(labels[feature], size=18)

    plt.tight_layout()
    plt.subplots_adjust()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_model_single(model, domain, fig_path=None):
    n_objectives = len(model.models)

    fig, axes = plt.subplots(nrows=1, ncols=n_objectives, figsize=(4*n_objectives,4))

    mean, var = model(domain)

    for ax, (feature, m) in zip(axes, model.models.items()):

        vmin, vmax = m.y.min(), m.y.max()
        tax = visualization.ternary_scatter_sub(domain.numpy(), mean[feature].numpy(), ax=ax, s=20, edgecolors=None, marker='H');
        ax.axis('equal')
        tax.scatter(m.X.numpy(), c=m.y.numpy(), s=30, edgecolors='k', cmap='Blues', vmin=vmin, vmax=vmax)

        ax.set_title(labels[feature], size=18)

    plt.tight_layout()
    plt.subplots_adjust()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_model_paneled(model, domain, fig_path=None):
    n_objectives = len(model.models)

    mean, var = model(domain)

    for feature, m in model.models.items():
        vmin, vmax = m.y.min(), m.y.max()
        tax = visualization.ternary_scatter(domain.numpy(), mean[feature].numpy(), s=20, edgecolors=None, marker='H', label=labels[feature]);
        tax.scatter(m.X.numpy(), c=m.y.numpy(), s=30, edgecolors='k', cmap='Blues', vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.subplots_adjust()

        if fig_path is not None:
            basename, ext = os.path.splitext(fig_path)
            plt.savefig(f'{basename}_{feature}{ext}', bbox_inches='tight')
            plt.clf()
            plt.close()

def plot_values(values, domain, fig_path=None):
    n_objectives = len(values)

    fig, axes = plt.subplots(nrows=1, ncols=n_objectives, figsize=(4*n_objectives,4))

    for ax, (feature, v) in zip(axes, values.items()):

        vmin, vmax = v.min(), v.max()
        tax = visualization.ternary_scatter_sub(domain.numpy(), v.numpy(), ax=ax, s=20, edgecolors=None, marker='H');
        # tax.scatter(m.X.numpy(), c=m.y.numpy(), s=30, edgecolors='k', cmap='Blues', vmin=vmin, vmax=vmax)
        ax.axis('equal')
        ax.set_title(labels[feature], size=18)

    plt.tight_layout()
    plt.subplots_adjust()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_pareto_set(values, grid, targets=None, sign=None, s=20, fig_path=None):
    """ plot the pareto set in the ternary simplex """
    if targets is None:
        targets = list(values.keys())

    # make sure to get the sign correct -- non_dominated_sort wants to minimize values
    Y = torch.stack(tuple(-sign[target]*values[target] for target in targets), dim=1).numpy()
    s, dominance = non_dominated_sort(Y)
    s_t = s[np.argsort(s[:,0])]

    visualization.ternary_scatter(grid.numpy(), dominance == 0, cmap='Reds', edgecolors=None, marker='h', label='pareto set')

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()

def compare_pareto(values, true_values, grid, targets=None, sign=None, fig_path=None):
    """ scatter plot of objective functions """

    if targets is None:
        targets = list(values.keys())

    k1, k2 = targets[:2]

    Y_t = torch.stack(tuple(-sign[target]*true_values[target] for target in targets), dim=1).numpy()
    s_t, dominance_t = non_dominated_sort(Y_t)
    _s_t = s_t[np.argsort(s_t[:,0])]

    # make sure to get the sign correct -- non_dominated_sort wants to minimize values
    Y = torch.stack(tuple(-sign[target]*values[target] for target in targets), dim=1).numpy()
    s, dominance = non_dominated_sort(Y)
    s_t = s[np.argsort(s[:,0])]

    sel = dominance == 0
    plt.scatter(true_values[k1].numpy()[~sel], true_values[k2].numpy()[~sel], c='k', alpha=0.1);
    plt.scatter(true_values[k1].numpy()[sel], true_values[k2].numpy()[sel], c='r', edgecolors='k', alpha=0.8);

    plt.plot(_s_t[:,0], _s_t[:,1], color='k', linestyle='--')
    plt.xlabel(f'true {labels[k1]}')
    plt.ylabel(f'true {labels[k2]}')
    # visualization.ternary_scatter(grid.numpy(), dominance == 0, cmap='Reds', edgecolors=None, marker='h')

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_measured_pareto_set(model, true_model, fig_path=None):
    # draw measured pareto plot...
    plt.figure(figsize=(4,4))
    k1, k2 = 'I_p', 'slope'

    # Y = torch.stack((model.models['I_p'].y, model.models['slope'].y), dim=1).numpy()

    # run the model to get mean predictions on observed data...
    mean, var = model(model.models[k1].X, drop_last=False)
    Y = torch.stack((mean[k1], mean[k2]), dim=1).numpy()
    s, dominance = non_dominated_sort(Y)
    s_t = s[np.argsort(s[:,0])]

    plt.plot(s_t[:,0], s_t[:,1], color='r', linestyle='--')
    sel = dominance == 0
    plt.scatter(Y[:,0][~sel], Y[:,1][~sel], c='k', alpha=0.25);
    plt.scatter(Y[:,0][sel], Y[:,1][sel], c='r', edgecolors='k', alpha=0.8);

    xerr = 1.96*var[k1].sqrt().numpy()
    yerr = 1.96*var[k2].sqrt().numpy()
    plt.errorbar(Y[:,0], Y[:,1], xerr=xerr, yerr=yerr, linestyle='none', color='k', zorder=-1)

    # true_mean, _ = em(_D)
    true_mean, _ = true_model.clean_iter_sample(None, uniform_noise=True)
    Y_t = torch.stack((true_mean[k1], true_mean[k2]), dim=1).numpy()
    s_t, dominance_t = non_dominated_sort(Y_t)
    _s_t = s_t[np.argsort(s_t[:,0])]

    plt.plot(_s_t[:,0], _s_t[:,1], color='k', linestyle='--')

    plt.xlabel(labels[k1])
    plt.ylabel(labels[k2])
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()
