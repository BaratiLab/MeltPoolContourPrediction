"""
Figures and reports for a trained cross-section model.

Every figure and summary is computed from the per-sample table returned by evaluation.evaluate().

Entry points:
    plot_loss_curve                training and validation loss from a wandb run
    plot_train_test_metrics        dataloader-level metrics, parity plots, error_metrics.json
    plot_metrics                   per-process-parameter (rollout) metrics and embedding maps
    plot_metrics_model_comparison  the rollout plots with one series per model

Metrics are defined in evaluation.py, contour geometry in utils.contour_utils and the parity plot
in utils.plotting_utils.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experimental_depth_prediction.analysis.result_analysis.evaluation import evaluate, summarize, METRICS
from experimental_depth_prediction.utils.contour_utils import contour_um
from experimental_depth_prediction.utils.plotting_utils import frame_tick, plot_parity

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGSIZE = np.array([4, 3]) * 1.3

# Parity-plot styling per quantity: plot units, axis limit, labels.
PARITY = {
    'area':  dict(scale = 1e-3, max_value = 65,
                  xlabel = r'Ground Truth Area $[\times 10^{3} \mu m^2]$', ylabel = r'Predicted Area $[\times 10^{3} \mu m^2]$'),
    'depth': dict(scale = 1, max_value = 450,
                  xlabel = r'Ground Truth Depth $[\mu m]$', ylabel = r'Predicted Depth $[\mu m]$'),
}

# Rollout plots: (metric, y label, column to color the points by, y limits, scale factor)
ROLLOUT_PLOTS = [
    ('iou', 'Mean IoU', 'power', (0, 1), 1),
    ('iou', 'Mean IoU', 'velocity', (0, 1), 1),
    ('area_percent_error', 'Mean Absolute Error [%]', 'velocity', None, 100),
    ('area_error', r'Mean Absolute Error [$\mu m^2$]', 'velocity', None, 1),
]
COLORBAR_TITLES = {'power': 'Power [W]', 'velocity': 'V [m/s]'}

# Embedding maps: for each column, (colorbar label, color limits)
EMBEDDING_COLOURS = {
    'energy_density': ('Energy Density [J/mm]', None),
    'depth_gt': (r'Ground Truth Depth [$\mu m$]', None),
    'depth_pred': (r'Predicted Depth [$\mu m$]', None),
    'depth_error': (r'Depth Error [$\mu m$]', (0, 30)),
    'depth_percent_error': ('Normalized Depth Error', (0, 1)),
}


# --------------------------------------------------------------------------- files and figures

def results_directory(run_path, subfolder, sweep_id = ''):
    """
    Return the results folder for a run, creating it if needed.

    Args:
        run_path (str): wandb run path. Its last component is the run id.
        subfolder (str): Folder inside the run's results folder.
        sweep_id (str): If given, results are written under results_<sweep_id> instead of results.

    Returns:
        str: <repo root>/results[_<sweep_id>]/<run id>/<subfolder>.
    """
    run_id = (run_path or 'unknown_result').split('/')[-1]
    results = f'results_{sweep_id}' if sweep_id else 'results'
    directory = REPO_ROOT / results / run_id / subfolder
    directory.mkdir(parents = True, exist_ok = True)
    return str(directory)


def finish(fig, path, show = False, formats = ('png',)):
    """
    Save a figure in each requested format, optionally show it, then close it.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        path (str): Output path without extension.
        show (bool): Whether to call plt.show() before closing.
        formats (tuple of str): File extensions to save.
    """
    for fmt in formats:
        fig.savefig(f'{path}.{fmt}', bbox_inches = 'tight')
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- figures

def plot_vs_energy_density(tables, metric, ylabel, color_by = None, scale = 1, ylim = None, title = None):
    """
    Plot the per-condition mean and standard deviation of a metric against energy density.

    Args:
        tables (dict): DataFrame from evaluate() for each series label. One series per entry.
        metric (str): Column to plot.
        ylabel (str): Y axis label.
        color_by (str, optional): 'power' or 'velocity'. With a single table the points are colored
            by this column with a colorbar. With several tables the series are distinguished by a
            legend instead.
        scale (float): Factor applied to the metric before plotting.
        ylim (tuple, optional): Y axis limits.
        title (str, optional): Figure title.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize = FIGSIZE, dpi = 150)
    colour = {}
    for label, df in tables.items():
        g = df.groupby('pv').agg(energy_density = ('energy_density', 'first'), power = ('power', 'first'),
                                 velocity = ('velocity', 'first'), mean = (metric, 'mean'), std = (metric, 'std'))
        colour = dict(c = g[color_by]) if color_by and len(tables) == 1 else {}
        points = ax.scatter(g.energy_density, scale * g['mean'], edgecolor = 'k', label = label, **colour)
        ax.errorbar(g.energy_density, scale * g['mean'], yerr = scale * g['std'], fmt = 'none', c = 'k', capsize = 3)
    if colour:
        fig.colorbar(points, ax = ax).ax.set_title(COLORBAR_TITLES[color_by])
    else:
        ax.legend()
    ax.set(xlabel = 'Energy Density [J/mm]', ylabel = ylabel, ylim = ylim, title = title)
    ax.grid()
    frame_tick(ax = ax)
    fig.tight_layout()
    return fig


def plot_iou_vs(df, column, xlabel, title = None):
    """
    Plot the IoU of every sample against one ground-truth column.

    Args:
        df (pandas.DataFrame): Table from evaluate().
        column (str): Column for the x axis, e.g. 'area_gt'.
        xlabel (str): X axis label.
        title (str, optional): Figure title.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize = np.array([4, 4]) * 1.3, dpi = 150)
    ax.scatter(df[column], df['iou'], edgecolors = 'k')
    ax.set(xlabel = xlabel, ylabel = 'IoU', ylim = (0, 1), title = title)
    frame_tick(ax = ax)
    fig.tight_layout()
    return fig


def plot_parity_figures(df, directory, partition, show = False):
    """Save area and depth parity plots, each as a raw scatter and as per-energy-density error bars."""
    for quantity, style in PARITY.items():
        for error_bars in (False, True):
            fig = plot_parity(df[f'{quantity}_gt'] * style['scale'], df[f'{quantity}_pred'] * style['scale'],
                              df['energy_density'], max_value = style['max_value'], band_style = 'absolute',
                              xlabel = style['xlabel'], ylabel = style['ylabel'], show_r2 = False,
                              alpha = 1 if error_bars else 0.05, error_bars = error_bars,
                              figsize = np.array([4, 4]) * 1.15, dpi = 600)
            prefix = 'error_bar_' if error_bars else ''
            finish(fig, os.path.join(directory, f'{prefix}{quantity}_{partition}'), show, formats = ('png', 'svg'))


def plot_embeddings(df, directory, stem):
    """Save PCA and t-SNE maps of the embedding column, one figure per coloring in EMBEDDING_COLOURS."""
    features = np.stack(df['embedding'].to_numpy())
    reducers = {'PCA': PCA(n_components = 2)}
    if len(features) >= 6:
        reducers['TSNE'] = TSNE(n_components = 2, perplexity = min(40, len(features) // 3))
    for name, reducer in reducers.items():
        low = reducer.fit_transform(features)
        for column, (label, bounds) in EMBEDDING_COLOURS.items():
            fig, ax = plt.subplots(figsize = np.array([4, 3]) * 1.6, dpi = 600)
            vmin, vmax = bounds or (None, None)
            points = ax.scatter(low[:, 0], low[:, 1], c = df[column], edgecolor = 'k', cmap = 'viridis', vmin = vmin, vmax = vmax)
            fig.colorbar(points, ax = ax, label = label)
            ax.set(xlabel = 'Dimension 1', ylabel = 'Dimension 2')
            frame_tick(ax = ax)
            finish(fig, os.path.join(directory, f'{stem}_{name}_{column}'), formats = ('png', 'svg'))


def plot_overlap(target, pred, directory, power, velocity):
    """Save a publication figure of the ground-truth and predicted outlines of one cross-section, in µm."""
    fig, ax = plt.subplots(figsize = np.array([4, 3]) * 1.2, dpi = 300)
    for cross_section, label in [(pred, 'Predicted'), (target, 'Ground Truth')]:
        outline = contour_um(cross_section)
        if outline is None:
            continue
        ax.fill(outline[:, 1], -outline[:, 0], alpha = 0.2)
        ax.plot(outline[:, 1], -outline[:, 0], linewidth = 2, label = label)
    ax.set(xlim = (0, 450), ylim = (-450, 0), aspect = 'equal', xlabel = r'Width $[\mu m]$', ylabel = r'Depth $[\mu m]$',
           title = f'$P$ = {int(power)} W, $V$ = {velocity} m/s')
    ax.legend(frameon = False, loc = 'lower right')
    frame_tick(ax = ax)
    finish(fig, os.path.join(directory, f'test_sample_contour_power_{int(power)}_velocity_{velocity}'), formats = ('png', 'svg'))


def plot_melt_pool(image, directory, power, velocity, frames = (0, 6, 12, 18, 24), spacing = 75, pixel_um = 5.6):
    """
    Save a publication figure of several raw thermal frames from one temporal-transformer sample.

    Args:
        image (numpy.ndarray): One input sequence as fed to the model, (S, C, H, W), in normalized
            counts.
        directory (str): Output folder.
        power, velocity: Process parameters, used in the title and file name.
        frames (tuple of int): Sequence indices to show, left to right.
        spacing (int): Pixel offset between consecutive frames in the composite.
        pixel_um (float): Camera pixel size in µm, for the axis extent.
    """
    panels = image[list(frames), 0].transpose(0, 2, 1) * 65536 / 10      # back to the camera's count scale
    height, width = panels.shape[1:]
    composite = np.full((height, spacing * (len(panels) - 1) + width), np.nan)
    for i, panel in enumerate(panels):
        composite[:, i * spacing:i * spacing + width] = panel
    composite[composite < 300] = np.nan

    fig, ax = plt.subplots(dpi = 300)
    extent = [0, composite.shape[1] * pixel_um, 0, height * pixel_um]
    im = ax.imshow(composite, vmin = 2000, vmax = 3500, cmap = 'inferno', extent = extent)
    fig.colorbar(im, cax = make_axes_locatable(ax).append_axes('right', size = '2%', pad = 0.05))
    stem = f'sample_image_power_{power}_velocity_{velocity}'
    for fmt in ('png', 'svg'):
        fig.savefig(os.path.join(directory, f'test_{stem}.{fmt}'), bbox_inches = 'tight')
    ax.set(xlabel = r'X [$\mu m$]', ylabel = r'Y [$\mu m$]', title = f'$P$ = {power} W, $V$ = {velocity} m/s')
    ax.grid()
    frame_tick(ax = ax)
    finish(fig, os.path.join(directory, f'test_axis_{stem}'), formats = ('png', 'svg'))


def plot_loss_curve(run, show = False, configuration_directory = None):
    """Save the training and validation loss per epoch from a wandb run's history."""
    configuration_directory = configuration_directory or os.path.join('results', run.id, 'config')
    history, config = run.history(), run.config
    fig, ax = plt.subplots(figsize = FIGSIZE, dpi = 150)
    for column, label in [('train_loss', 'Train'), ('test_loss', 'Test')]:
        ax.plot(history[column].dropna().to_numpy(), label = label)
    ax.set(xlabel = 'Epoch', ylabel = 'Loss',
           title = f'Sequence Length: {config["seq_length"]}, Model: {config["model_type"]}, Sequence Type: {config["sequence_type"]}')
    ax.legend()
    ax.grid()
    frame_tick(ax = ax)
    finish(fig, os.path.join(configuration_directory, 'loss'), show)


# --------------------------------------------------------------------------- entry points

def plot_train_test_metrics(model, train_dataloader, test_dataloader, run_path = None, show = False, img_shape = None,
                            sweep_id = '', config = None, plot_publication_figures = False, partitions = ('Test',)):
    """
    Evaluate one run on whole dataloaders and save the summary figures and metrics.

    For each partition this saves the per-sample table (summary/samples_<partition>.csv), the area and
    depth parity plots and IoU-vs-size plots (plot_train_test/), and the metric summary of all
    partitions as summary/error_metrics.json. With plot_publication_figures, the first 30 test
    samples also get an outline-overlap figure (and a raw-frame composite for temporal transformers).

    Returns:
        dict: {partition: summarize(df)}.
    """
    method = 'SDF' if config['sdf'] else 'binary'
    plotting_directory = results_directory(run_path, 'plot_train_test', sweep_id)
    summary_directory = results_directory(run_path, 'summary', sweep_id)
    loaders = {'Train': train_dataloader, 'Test': test_dataloader}

    error_metrics = {}
    for partition in partitions:
        n_keep = 30 if plot_publication_figures and partition == 'Test' else 0
        df = evaluate(model, loaders[partition], method, img_shape, n_keep = n_keep)
        df.to_csv(os.path.join(summary_directory, f'samples_{partition}.csv'), index = False)
        error_metrics[partition] = summarize(df)

        plot_parity_figures(df, plotting_directory, partition, show)
        for column, xlabel in [('area_gt', r'Ground Truth Area [$\mu m^2$]'), ('depth_gt', r'Ground Truth Depth [$\mu m$]')]:
            fig = plot_iou_vs(df, column, xlabel, title = f'{partition} Set IoU')
            finish(fig, os.path.join(plotting_directory, f'{column.split("_")[0]}_iou_{partition}'), show)
        for sample in df.attrs['samples']:
            plot_overlap(sample['target'], sample['prediction'], plotting_directory, sample['power'], sample['velocity'])
            if config['model_type'] == 'temp_transformer':
                plot_melt_pool(sample['image'], plotting_directory, sample['power'], sample['velocity'])

    with open(os.path.join(summary_directory, 'error_metrics.json'), 'w') as f:
        json.dump(error_metrics, f, indent = 2)
    return error_metrics


def plot_metrics(dataset, partition, model, model_type, run_path = None, show = False, img_shape = None,
                 sweep_id = '', embeddings = True):
    """
    Evaluate one dataset split sample by sample and save the per-process-parameter (rollout) figures.

    Saves the per-sample table and a per-PV mean/std table (metrics/), the metric-vs-energy-density
    plots and a depth parity plot (plot_metrics/) and, when embeddings = True and the model type
    exposes a latent, PCA / t-SNE maps of the embeddings (dim_reduction/).

    Returns:
        dict: summarize(df) with every key prefixed 'rollout_'.
    """
    method = 'SDF' if dataset.sdf else 'binary'
    metrics_directory = results_directory(run_path, 'metrics', sweep_id)
    plotting_directory = results_directory(run_path, 'plot_metrics', sweep_id)

    df = evaluate(model, DataLoader(dataset, batch_size = 1, shuffle = False), method, img_shape,
                  model_type = model_type, embeddings = embeddings)
    table = df.drop(columns = 'embedding', errors = 'ignore')
    table.to_csv(os.path.join(metrics_directory, f'samples_{partition}.csv'), index = False)
    table.groupby('pv')[METRICS].agg(['mean', 'std']).to_csv(os.path.join(metrics_directory, f'per_pv_{partition}.csv'))

    title = f'{partition.capitalize()} Set'
    for metric, ylabel, color_by, ylim, scale in ROLLOUT_PLOTS:
        fig = plot_vs_energy_density({partition: df}, metric, ylabel, color_by = color_by, scale = scale, ylim = ylim, title = title)
        finish(fig, os.path.join(plotting_directory, f'rollout_energy_density_{metric}_by_{color_by}_{partition}'), show)
    style = PARITY['depth']
    fig = plot_parity(df['depth_gt'], df['depth_pred'], df['energy_density'], title = f'{title}, Sequence Rollout',
                      max_value = style['max_value'], band_style = 'absolute', xlabel = style['xlabel'], ylabel = style['ylabel'])
    finish(fig, os.path.join(plotting_directory, f'rollout_energy_density_depth_{partition}'), show)

    if 'embedding' in df:
        plot_embeddings(df, results_directory(run_path, 'dim_reduction', sweep_id), stem = f'rollout_{partition}_{model_type}')
    return {f'rollout_{key}': value for key, value in summarize(df).items()}


def plot_metrics_model_comparison(dataset_dict, partition, model_dict, config, legend_dict = None, show = False, img_shape = None):
    """
    Save the rollout metric-vs-energy-density plots with one series per model.

    Args:
        dataset_dict, model_dict (dict): the dataset and model for each model name.
        config (dict): shared run config; only 'sdf' is used.
        legend_dict (dict, optional): display label for each model name.

    Saves each model's per-sample table under results/comparison/<model name>/metrics/ and the
    plots under results/comparison/plot_metrics/.
    """
    method = 'SDF' if config['sdf'] else 'binary'
    legend_dict = legend_dict or {}
    tables = {}
    for name, model in model_dict.items():
        df = evaluate(model, DataLoader(dataset_dict[name], batch_size = 1, shuffle = False), method, img_shape)
        df.to_csv(os.path.join(results_directory(f'comparison/{name}', 'metrics'), f'samples_{partition}.csv'), index = False)
        tables[legend_dict.get(name, name)] = df

    plotting_directory = results_directory('comparison', 'plot_metrics')
    for metric, ylabel, _, ylim, scale in ROLLOUT_PLOTS[1:]:
        fig = plot_vs_energy_density(tables, metric, ylabel, scale = scale, ylim = ylim, title = f'{partition.capitalize()} Set')
        finish(fig, os.path.join(plotting_directory, f'energy_density_{metric}_{partition}'), show)
    return tables
