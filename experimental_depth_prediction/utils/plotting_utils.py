
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
from sklearn.metrics import r2_score


def frame_tick(frame_width = 2, tick_width = 1.5, ax = None):
    """
    Thicken the frame and point the ticks inwards.

    Args:
        frame_width (float): line width of the frame.
        tick_width (float): width of the tick marks.
        ax (matplotlib.axes.Axes, optional): axes to style. Defaults to the current axes.
    """
    ax = ax if ax is not None else plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(frame_width)
    ax.tick_params(direction = 'in', width = tick_width)


def legend(location = 'upper left', fontsize = 8):
    """Add a legend without a frame to the current axes."""
    plt.legend(loc = location, fontsize = fontsize, frameon = False)
    
def savefig(filename):
    """Save the current figure with a tight bounding box."""
    plt.savefig(filename, bbox_inches = 'tight')



def log_figure(fig, name, epoch, results_folder = None):
    """
    Upload a matplotlib figure to wandb, optionally save it as a PNG, then close it.

    Args:
        fig (matplotlib.figure.Figure): The figure that is logged and optionally saved.
        name (str): The filename (<name>.png), also used as the wandb key for the image.
        epoch (int): wandb step for logging the image.
        results_folder (str, optional): folder to save <name>.png in. The file is overwritten on disk
            each time it is logged, so only the latest version is saved locally. Wandb stores every epoch.
    """
    if results_folder is not None:
        fig.savefig(os.path.join(results_folder, f'{name}.png'), bbox_inches = 'tight')
    wandb.log({name: wandb.Image(fig)}, step = epoch)
    plt.close(fig)


def plot_cross_section(image, title):
    """One cross-section (label or prediction) as an image."""
    fig, ax = plt.subplots(figsize = (5.2, 3.9), dpi = 150)
    ax.imshow(image)
    ax.set(xlabel = 'X', ylabel = 'Y', title = title)
    frame_tick(ax = ax)  

    return fig


def plot_parity(true_values, pred_values, energy_density, title = None, max_value = 70000, band = 0.1,
                band_style = 'relative', xlabel = r'Ground Truth Area [$\mu m^2$]',
                ylabel = r'Predicted Area [$\mu m^2$]', show_r2 = True, alpha = 1, error_bars = False,
                figsize = (5.2, 5.2), dpi = 150):
    """
    Plot predicted against true values, colored by energy density, with a band around the diagonal.

    Args:
        true_values, pred_values (array, (N,)): The quantity being compared, already in plot units.
        energy_density (array, (N,)): power divided by velocity in J/mm, used for the color on a fixed 0 to 1.5 scale.
        title (str, optional): Figure title.
        max_value (float): Axis limit. Both axes run from 0 to max_value.
        band (float): Half-width of the dashed lines around the diagonal. With band_style 'relative'
            they sit at (1 ± band) * true, with 'absolute' at true ± band * max_value.
        band_style (str): 'relative' or 'absolute'.
        xlabel, ylabel (str): Axis labels.
        show_r2 (bool): Whether to write the R² of the raw samples in the corner.
        alpha (float): Marker opacity.
        error_bars (bool): Whether to plot one point per energy density, the mean prediction ± std
            against the mean true value, instead of every sample.
        figsize, dpi: Passed to plt.subplots.

    Returns:
        matplotlib.figure.Figure
    """
    true_values, pred_values, energy_density = (np.asarray(a, dtype = float)
                                                for a in (true_values, pred_values, energy_density))
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

    if error_bars:
        levels = np.unique(energy_density)
        groups = [(true_values[energy_density == e], pred_values[energy_density == e]) for e in levels]
        x = np.array([t.mean() for t, _ in groups])
        y = np.array([p.mean() for _, p in groups])
        ax.errorbar(x, y, yerr = [p.std() for _, p in groups], fmt = 'none', ecolor = 'k', capsize = 3)
        colours = levels
    else:
        x, y, colours = true_values, pred_values, energy_density
    points = ax.scatter(x, y, c = colours, vmin = 0, vmax = 1.5, edgecolors = 'k', alpha = alpha)
    fig.colorbar(points, ax = ax, fraction = 0.046, pad = 0.04).ax.set_title('Energy Density [J/mm]')

    diagonal = np.array([0, max_value])
    offset = band * diagonal if band_style == 'relative' else band * max_value
    ax.plot(diagonal, diagonal, 'k-')
    for sign in (-1, 1):
        ax.plot(diagonal, diagonal + sign * offset, 'k--')
    if show_r2:
        ax.text(0.05, 0.95, f'$R^2$ = {r2_score(true_values, pred_values):.2f}', transform = ax.transAxes, va = 'top')
    ax.set(xlabel = xlabel, ylabel = ylabel, title = title, xlim = (0, max_value), ylim = (0, max_value), aspect = 'equal')
    frame_tick(ax = ax)
    return fig


def plot_epoch(results, epoch, pixel_area, results_folder = None):
    """
    Log the monitoring plots for one epoch.

    For each split this logs one sample's true and predicted cross-section and a parity plot of
    predicted against true melt pool area, under the keys <split>_img_target, <split>_img_output and
    <split>_area_epoch.

    Args:
        results (dict): Per-split results, e.g. {'train': {...}, 'test': {...}}. Each value needs:
            'labels' (array, (N, H, W)): Label cross-sections. The first is plotted.
            'predictions' (array, (N, H, W)): Predicted cross-sections. The first is plotted.
            'true_areas', 'pred_areas' (array, (N,)): Melt pool areas in µm².
            'energy_density' (array, (N,)): power divided by velocity in J/mm, used to color the parity plot.
        epoch (int): Current epoch, used in titles and as the wandb step.
        pixel_area (float): Area of one label pixel in µm².
        results_folder (str, optional): Folder to save the PNGs in. Nothing is saved to disk if None.
    """
    for split, r in results.items():
        for kind, image in [('target', r['labels'][0]),
                            ('output', r['predictions'][0])]:
            title = f'{split.capitalize()} Example, epoch {epoch}, area = {image.sum() * pixel_area:.0f} µm²'

            log_figure(plot_cross_section(image, title), 
                       f'{split}_img_{kind}', 
                       epoch, 
                       results_folder)

        log_figure(plot_parity(r['true_areas'], 
                               r['pred_areas'], 
                               r['energy_density'], 
                               f'{split.capitalize()} Set Area'),
                               f'{split}_area_epoch', epoch, results_folder)