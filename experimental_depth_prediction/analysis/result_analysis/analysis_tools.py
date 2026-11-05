import os
from matplotlib.animation import FuncAnimation
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import seaborn as sns
import sklearn
from experimental_depth_prediction.utils.utils import compute_iou
from experimental_depth_prediction.utils.plotting_utils import frame_tick, savefig
import json
import seaborn as sns
from skimage import measure
from experimental_depth_prediction.utils.utils import sdf_to_binary, resample_sdf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d


import matplotlib.pyplot as plt
import os
import seaborn as sns

device = 'cuda'

IMAGE_EXTENT_X = 368.208
IMAGE_EXTENT_Y = 450.65


def reconstruct(t,coeffs, verbose = False, device = 'cuda'):
    order = (coeffs.shape[1])//2 - 1
    x_coeffs = coeffs[:,:(coeffs.shape[1])//2]
    y_coeffs = coeffs[:,(coeffs.shape[1])//2:]
    result_x = torch.zeros(t.shape).float().to(device)
    result_y = torch.zeros(t.shape).float().to(device)

    for idx,i in enumerate(range(order + 1)):
        degree = order - idx
        if verbose:
            print(degree)
            print(result_x)
            print(y_coeffs.shape)
        result_x += x_coeffs[:,idx:idx+1]*t**degree
        if verbose:
            print(y_coeffs[:,idx:idx+1].shape, t.shape)
        result_y += y_coeffs[:,idx:idx+1]*t**degree
    if torch.any(torch.isinf(result_x)):
        breakpoint()

    return result_x, result_y

def create_results_directory(run_path, subfolder, sweep_id = ''):
    """
    Creates a directory for the results of a run.

    Parameters:
    - run_path: The path to the run.
    - subfolder: The subfolder to create.
    Returns:
    - str: The path to the created directory.
    """
    if run_path is None:
        run_path = 'unknown_result'
    current_directory = os.path.dirname(os.path.realpath(__file__))
    results_directory = os.path.join(current_directory, '../../../', f'results_{sweep_id}', run_path.split('/')[-1], subfolder)
    os.makedirs(results_directory, exist_ok=True)
    return results_directory

def plot_loss_curve(run_object, show=False, configuration_directory = None):
    """
    Plots the loss curve for a given run.

    Parameters:
    - run: The run object containing the loss history.
    - show: Boolean value indicating whether to display the plot or save it as an image.
    - configuration_directory: The directory to save the plot to.
    Returns:
    None
    """
    if configuration_directory is None:
        configuration_directory = os.path.join('results', run_object.id, 'config')
    sns.reset_orig()
    df = run_object.history()
    config = run_object.config
    plt.figure(figsize=np.array([4, 3]) * 1.3, dpi=150)
    plt.title(f'Sequence Length: {config["seq_length"]}, Model: {config["model_type"]}, Sequence Type: {config["sequence_type"]}')
    # Train Loss
    filtered_train_df = df[df['train_loss'].notnull()]
    plt.plot(np.array(filtered_train_df['train_loss']), label='Train')
    plt.xlim([0, len(filtered_train_df['_step']) - 1])
    # Test Loss
    filtered_df = df[df['test_loss'].notnull()]
    plt.plot(np.array(filtered_df['test_loss']), label='Test')
    plt.legend()
    plt.grid()
    frame_tick()
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(configuration_directory, 'loss.png'), bbox_inches='tight')
    plt.clf()


def plot_contour_depth_area(pv_metrics, highlight_pv, plot = 'area', ax  = None, show = False, plotting_directory = None):


    areas = pv_metrics['areas_gt']
    areas_pred = pv_metrics['areas_pred'][highlight_pv] 
    depths = pv_metrics['depths_gt']
    depths_pred = pv_metrics['depths_pred'][highlight_pv] 
    density_list = []
    area_list = []
    depth_list = []
    for key in areas.keys():
        if 'power' in key:
            power = float(key.split('power')[-1].split('_')[0])
            velocity = float(key.split('velocity')[-1].split('_')[0])
        else:
            power = float(key.split('_')[0])
            velocity = float(key.split('_')[1])
        density = float(power)/float(velocity)
        density_list.append(density)
        area_list.append(np.mean(areas[key]))
        depth_list.append(np.mean(depths[key]))

    density_list = np.array(density_list)
    area_list = np.array(area_list)
    depth_list = np.array(depth_list)
    if highlight_pv is not None:
        if 'power' in highlight_pv:
            highlight_power = float(highlight_pv.split('power')[-1].split('_')[0])
            highlight_velocity = float(highlight_pv.split('velocity')[-1].split('_')[0])
        else:
            highlight_power = float(highlight_pv.split('_')[0])
            highlight_velocity = float(highlight_pv.split('_')[1])
    pred_density = float(highlight_power/highlight_velocity)
    gt_density = float(highlight_power/highlight_velocity)
    
    for highlight_time in tqdm(range(len(areas_pred))):
        # if ax is None:
        plt.figure(dpi = 150, figsize = np.array([4, 3])*1.3)
        ax = plt.gca()

        pred_area = areas_pred[highlight_time]
        pred_depth = depths_pred[highlight_time]
        gt_area = areas[highlight_pv][highlight_time]
        gt_depth = depths[highlight_pv][highlight_time]
        plot_folder = os.path.join(plotting_directory, f'contour_{plot}')
        os.makedirs(plot_folder, exist_ok = True)
        if plot == 'area':
            ax.scatter(density_list, area_list, alpha = 0.5)
            ax.scatter(pred_density, pred_area, color = 'r', label = 'Prediction')
            ax.scatter(gt_density, gt_area, color = 'g', label = 'Ground Truth') 
            ax.set_xlabel('Energy Density (J/mm)')
            ax.set_ylabel(r'Contour Area ($\mu m^2$)')
            # ax.set_title('Contour Area vs Energy Density')
            frame_tick()
            ax.legend()
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(plot_folder, 'contour_area_{:03d}.png'.format(highlight_time)) , bbox_inches = 'tight')
                plt.clf()
        elif plot == 'depth':
            ax.scatter(density_list, depth_list, alpha = 0.5)
            ax.scatter(pred_density, pred_depth, color = 'r', label = 'Prediction')
            ax.scatter(gt_density, gt_depth, color = 'g', label = 'Ground Truth')
            ax.set_xlabel('Energy Density (J/mm)')
            ax.set_ylabel(r'Contour Depth ($\mu m$)')
            ax.set_title('Contour Depth vs Energy Density')
            frame_tick()
            ax.legend()
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(plot_folder, 'contour_depth_{:03d}.png'.format(highlight_time)), bbox_inches = 'tight')
                plt.clf()
        
        plt.close('all')

def get_label_from_model_name(legend_dict, model_name):
    """
    Retrieves the label associated with a given model name from the legend dictionary.
    
    Parameters:
        legend_dict (dict): A dictionary containing model names as keys and their corresponding labels as values.
        model_name (str): The name of the model for which the label is to be retrieved.
        
    Returns:
        str: The label associated with the given model name. If the legend dictionary is None, returns the model name itself.
    """
    if legend_dict is None:
        label = model_name
    else:
        label = legend_dict[model_name] if model_name in legend_dict else model_name
    return label


def plot_metrics_model_comparison(dataset_dict, partition, model_dict, config, debug_plot=False, run_path=None, legend_dict=None, show=False, create_videos = False):
    """
    Plot the metrics for model comparison.

    Args:
        dataset_dict (dict): A dictionary containing the datasets for each model.
        partition (str): The partition of the dataset ('train', 'val', or 'test').
        model_dict (dict): A dictionary containing the models.
        debug_plot (bool, optional): Whether to enable debug plots. Defaults to False.
        run_path (str, optional): The path to the run. Defaults to None.
        legend_dict (dict, optional): A dictionary containing the legends for each model. Defaults to None.
        show (bool, optional): Whether to show the plots. Defaults to False.
    """
    model_comparison_results = {}

    for model_name in model_dict.keys():
        dataset = dataset_dict[model_name]
        model = model_dict[model_name]
        plotting_directory = os.path.join('results', 'comparison', model_name, 'plot_metrics')
        metrics_directory = os.path.join('results', 'comparison', model_name, 'metrics')
        
        print('metrics saved here')
        os.makedirs(metrics_directory, exist_ok=True)
        os.makedirs(plotting_directory, exist_ok=True)
        print(metrics_directory)
        
        items_info_pv, items_pv, items_info, items = dataset.process_parameter_mapping()
        information = {
        'exposure': dataset.exposure,
        'image_type': dataset.image_type,
        'sequence_type': dataset.sequence_type,
        'sequence_length': dataset.sequence_length
        }
        pv_metrics, pv_listed_metrics = plot_results(dataset=dataset,
                                                model=model,
                                                partition=partition,
                                                model_type=config['model_type'],
                                                debug_plot=debug_plot,
                                                plot=False,
                                                show=show,
                                                run_path=run_path,
                                                folder='original', create_videos=create_videos)
        
       
        pv_ious = pv_metrics['ious']
        pv_mean_ious = pv_metrics['mean_iou']
        pv_area_maes = pv_metrics['area_maes']
        pv_areas_gt = pv_metrics['areas_gt']
        pv_areas_pred = pv_metrics['areas_pred']
        pv_depths_gt = pv_metrics['depths_gt']
        pv_depths_pred = pv_metrics['depths_pred']
       
        with open(os.path.join(metrics_directory, 'pv_ious.json'), 'w') as f:
            json.dump(pv_ious, f)
        with open(os.path.join(metrics_directory, 'pv_mean_ious.json'), 'w') as f:
            json.dump(pv_mean_ious, f)
        with open(os.path.join(metrics_directory, 'pv_area_maes.json'), 'w') as f:
            json.dump(pv_area_maes, f)
        with open(os.path.join(metrics_directory, 'pv_areas_gt.json'), 'w') as f:
            json.dump(pv_areas_gt, f)
        with open(os.path.join(metrics_directory, 'pv_areas_pred.json'), 'w') as f:
            json.dump(pv_areas_pred, f)
        with open(os.path.join(metrics_directory, 'pv_depths_gt.json'), 'w') as f:
            json.dump(pv_depths_gt, f)
        with open(os.path.join(metrics_directory, 'pv_depths_pred.json'), 'w') as f:
            json.dump(pv_depths_pred, f)
        
        plt.close('all')
    
        model = model_dict[model_name]
        comparison_results = {}
        comparison_results['mean_iou'] = []
        comparison_results['e_densities'] = []
        comparison_results['std_iou'] = []
        comparison_results['powers'] = []
        comparison_results['velocities'] = []
        comparison_results['area_maes'] = []
        comparison_results['area_std_maes'] = []
        comparison_results['area_raw_maes'] = []
        comparison_results['area_raw_std_maes'] = []
    
        for key in pv_mean_ious.keys():
            p, v = key.split('_')
            comparison_results['powers'].append(int(p))
            comparison_results['velocities'].append(float(v))
            energy_density = float(p) / (1000 * float(v))
            comparison_results['e_densities'].append(energy_density)
            comparison_results['mean_iou'].append(pv_mean_ious[key])
            comparison_results['std_iou'].append(np.std(pv_ious[key]))
            comparison_results['area_maes'].append(np.mean(pv_area_maes[key]))
            comparison_results['area_std_maes'].append(np.std(pv_area_maes[key]))
            comparison_results['area_raw_maes'].append(np.mean(np.abs(np.array(pv_areas_gt[key]) - np.array(pv_areas_pred[key]))))
            comparison_results['area_raw_std_maes'].append(np.std(np.abs(np.array(pv_areas_gt[key]) - np.array(pv_areas_pred[key]))))
        
        model_comparison_results[model_name] = comparison_results
        
    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    for model_name in model_dict.keys():
        label = get_label_from_model_name(legend_dict, model_name)
        plt.scatter(model_comparison_results[model_name]['e_densities'], model_comparison_results[model_name]['mean_iou'], edgecolor='k', label=label)
        plt.errorbar(model_comparison_results[model_name]['e_densities'], model_comparison_results[model_name]['mean_iou'], yerr=model_comparison_results[model_name]['std_iou'], fmt='none', c='k', capsize=3)
    
    clb = plt.colorbar()
    clb.ax.set_title("Power [W]")
    plt.xlabel(r'Energy Density [$J/{m^3}$]')
    plt.ylabel('Mean IoU')
    plt.ylim([0, 1])
    plt.legend()
    frame_tick()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'energy_density_iou_{partition}.png'))

    if show:
        plt.show()
    else:
        plt.clf()

    plt.close('all')
    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    for model_name in model_dict.keys():
        label = get_label_from_model_name(legend_dict, model_name)
        plt.scatter(model_comparison_results[model_name]['e_densities'], model_comparison_results[model_name]['mean_iou'], edgecolor='k', label=label)
        plt.errorbar(model_comparison_results[model_name]['e_densities'], model_comparison_results[model_name]['mean_iou'], yerr=model_comparison_results[model_name]['std_iou'], fmt='none', c='k', capsize=3)
  
    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean IoU')
    plt.ylim([0, 1])
    frame_tick()
    plt.legend()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'energy_density_iou_{partition}.png'))

    if show:
        plt.show()
    else:
        plt.clf()



    plt.figure(dpi = 150, figsize= np.array([4,3])*1.3)
    for model_name in model_dict.keys():
        label = get_label_from_model_name(legend_dict, model_name)
        plt.scatter(model_comparison_results[model_name]['e_densities'], 100*np.array(model_comparison_results[model_name]['area_maes']), edgecolor = 'k', label=label)
        plt.errorbar(model_comparison_results[model_name]['e_densities'], 100*np.array(model_comparison_results[model_name]['area_maes']), yerr = 100*np.array(model_comparison_results[model_name]['area_std_maes']), fmt = 'none', c = 'k', capsize=3)

 
    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean Absolute Error [%]')
    # plt.ylim([0, 1])

    frame_tick()
    title_partition = partition[0].upper() + partition[1:]
    plt.legend()
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'energy_density_area_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()


    plt.close('all')
    plt.figure(dpi = 150, figsize= np.array([4,3])*1.3)
    for model_name in model_dict.keys():
        label = get_label_from_model_name(legend_dict, model_name)
        plt.scatter(model_comparison_results[model_name]['e_densities'], np.array(model_comparison_results[model_name]['area_raw_maes']), edgecolor = 'k', label=label)
        plt.errorbar(model_comparison_results[model_name]['e_densities'], np.array(model_comparison_results[model_name]['area_raw_maes']), yerr = np.array(model_comparison_results[model_name]['area_raw_std_maes']), fmt = 'none', c = 'k', capsize=3)
    

    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean Absolute Error [$\mu m^2$]')
    # plt.ylim([0, 1])
    frame_tick()
    plt.legend()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'energy_density_area_raw_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    plt.close('all')


def plot_dimensionality_reduction(results, labels, title, dim_reduction_directory, text, cbar_title, figure_fname, color_map='viridis', xlabel='Dimension 1', ylabel='Dimension 2', cbounds = None):
    """
    Plots the results of dimensionality reduction.

    Parameters:
    - results (array-like): The dimensionally reduced data.
    - labels (array-like): Labels or values for color coding the points in the plot.
    - model_type (str): The type of model used for training.
    - title (str): The title of the plot.
    - color_map (str, optional): The color map for the plot. Defaults to 'viridis'.
    - xlabel (str, optional): Label for the X-axis. Defaults to 'Dimension 1'.
    - ylabel (str, optional): Label for the Y-axis. Defaults to 'Dimension 2'.
    """
    replacements = {
        'GT Depth': 'Experimental Depth [$\mu m$]',
        'Predicted Depth': 'Predicted Depth [$\mu m$]',
        'E Density': "Energy Density [J/mm]"
    }
    # breakpoint()
    sns.set(style = 'ticks')
    plt.figure(dpi = 600, figsize = np.array([4,3])*1.6)
    # breakpoint()
    if cbounds is None:
        plt.scatter(results[:,0], results[:,1], c = labels, edgecolor = 'k', cmap = 'viridis')
    else:
        plt.scatter(results[:,0], results[:,1], c = labels, edgecolor = 'k', vmin = cbounds[0], vmax = cbounds[1],cmap = 'viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(f'{title}')
    ax = plt.gca()
    # plt.text(0.05,0.94, text, transform = ax.transAxes, fontsize = 8)
    frame_tick()
    plt.colorbar(label = cbar_title)
    # clb.ax.set_title(cbar_title, fontsize = 8)

    savefig(os.path.join(dim_reduction_directory, f'{figure_fname}.png'))
    savefig(os.path.join(dim_reduction_directory, f'{figure_fname}.svg'))

    plt.clf()

def plot_dimensionality_reduction_video(pc_model, results, labels, pv_embeddings, highlighted_pv, title, dim_reduction_directory, text, cbar_title, figure_fname, information, color_map='viridis', xlabel='Dimension 1', ylabel='Dimension 2', run_path = None):
    """
    Plots the results of dimensionality reduction.

    Parameters:
    - results (array-like): The dimensionally reduced data.
    - labels (array-like): Labels or values for color coding the points in the plot.
    - model_type (str): The type of model used for training.
    - title (str): The title of the plot.
    - color_map (str, optional): The color map for the plot. Defaults to 'viridis'.
    - xlabel (str, optional): Label for the X-axis. Defaults to 'Dimension 1'.
    - ylabel (str, optional): Label for the Y-axis. Defaults to 'Dimension 2'.
    """
    exposure = information['exposure']
    image_type = information['image_type']
    sequence_type = information['sequence_type']
    sequence_length = information['sequence_length']
    results_specific = pc_model.transform(np.array(pv_embeddings[highlighted_pv]).reshape(len(pv_embeddings[highlighted_pv]), -1)) #pc_model.transform(np.array(pv_embeddings[highlighted_pv]))
    
    fig = plt.figure(dpi = 150, figsize = np.array([4,3])*1.3, facecolor='white')
    os.makedirs(os.path.join(dim_reduction_directory, 'pca_animation'), exist_ok=True)
    plot_folder = f'result_new_plots/thermal_images/{highlighted_pv}/{image_type}/{sequence_type}/{sequence_length}/{exposure}'
    os.makedirs(plot_folder, exist_ok=True)
    for index in tqdm(range(len(results_specific))):
        plt.clf()
        ax = plt.gca()
        plt.scatter(results[:,0], results[:,1], c = labels, edgecolor = 'k')
        plt.xlabel(xlabel)
        plt.xlim([0,1])
        plt.ylabel(ylabel)
        plt.title('Frame: {:03d}'.format(index), fontsize = 8)

        # plt.title(f'{title}' + ', Frame: {:03d}'.format(index), fontsize = 8)
        plt.text(0.05,0.94, text, transform = ax.transAxes, fontsize = 8)
        frame_tick()
        clb = plt.colorbar()
        clb.ax.set_ylabel(cbar_title + ' [$\mu m$]', fontsize = 8)
        plt.scatter(results_specific[index,0], results_specific[index,1], c = 'r', edgecolor = 'k')
        plt.savefig(os.path.join(dim_reduction_directory, 'pca_animation', 'frame_{:03d}.png'.format(index)))
        plt.savefig(os.path.join(plot_folder, 'frame_{:03d}.png'.format(index)))
    plt.close('all')

def plot_data(sample_image, prediction, target, frame_value, power, velocity, plotting_directory, title, fig, axs, show=False):
    """
    Plots the sample image, model prediction, and ground truth.

    Parameters:
    - sample_image: The sample image.
    - prediction: The prediction made by the model.
    - target: The ground truth data.
    - frame_value: Frame number or identifier.
    - power: Power parameter for the title.
    - velocity: Velocity parameter for the title.
    - plotting_directory: Directory to save the plots.
    - show: Boolean flag to indicate whether to display the plot or save it.
    """

    axs[0].imshow(sample_image[0][0].cpu().detach().numpy(), cmap = 'inferno')
    # img_shape = int(np.round(np.sqrt(sample_image.shape[-1])))
    if np.log2(sample_image.shape[-1]) % 2 != 0:
        img_shape = int(np.round(np.sqrt(sample_image.shape[-1]/2)))
    else:
        img_shape = int(np.round(np.sqrt(sample_image.shape[-1])))
    axs[1].imshow(prediction.reshape(prediction.shape[0], 2, img_shape,img_shape)[0].argmax(dim = 0).cpu().detach().numpy())
    axs[2].imshow(target[0].cpu().detach().numpy())

    # axs[0].set_title(f'[Frame {frame_value}] -> [Frame {frame_value + len(items_pv[pv_combo][i][0])}]')
    axs[0].set_title(title)
    axs[1].set_title('Prediction')
    axs[2].set_title('Ground Truth')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    fig.suptitle(f'Test Prediction, Power = {power} W, Velocity = {velocity} m/s', y = 0.8)
    fig.tight_layout()


    if show:
        plt.show()
    else:
        # Saving the figure
        plot_filename = 'plot_frame_{:05}.png'.format(frame_value)
        plot_path = os.path.join(plotting_directory, plot_filename)
        fig.savefig(plot_path)
        plt.close(fig)

def plot_thermal_image_contour_over_time(pv_thermal_image, pv, pv_contours, information, model, contour_gt = None, contour_pred = None, plotting_directory = None, img_shape = None):
    exposure = information['exposure']
    image_type = information['image_type']
    sequence_type = information['sequence_type']
    sequence_length = information['sequence_length']
    repeat = information['repeat']
    power = float(pv.split('_')[0])
    velocity = float(pv.split('_')[1])
    areas = []
    depths = []
    predicted_areas = []
    predicted_depths = []

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi = 150, facecolor='white')
    os.makedirs(os.path.join(plotting_directory, 'animation_with_contour'), exist_ok=True)
    for i in pv_contours[pv]:
        depth = np.max(np.where(i)[0])#np.max(np.where(i))
        area = i.sum()
        areas.append(area)
        depths.append(depth)
    # Function to update the figure for each frame
    for index in tqdm(range(len(pv_thermal_image[pv]))):

        if len(pv_thermal_image[pv].shape) < 5 and repeat:
            input_image = np.repeat(pv_thermal_image[pv][index][None], 3, axis = 0)[None]
            plotting_image = pv_thermal_image[pv][index]
        elif len(pv_thermal_image[pv].shape) < 5:
            input_image = pv_thermal_image[pv][index][None][None]
            plotting_image = pv_thermal_image[pv][index]
        else:
            input_image = pv_thermal_image[pv][index][None]
            plotting_image = pv_thermal_image[pv][index][0][0]
        prediction = model(torch.tensor(input_image).float().to(device))
        # img_shape = int(np.round(np.sqrt(prediction.shape[-1])))
        if img_shape is None:
            if np.log2(prediction.shape[-1]) % 2 != 0:
                img_shape = int(np.round(np.sqrt(prediction.shape[-1]/2)))
            else:
                img_shape = int(np.round(np.sqrt(prediction.shape[-1])))
        contour_pred = prediction.reshape(prediction.shape[0], 2, img_shape,img_shape)[0].argmax(dim = 0).cpu().detach().numpy()
        area = contour_pred.sum()
        depth = np.max(np.where(contour_pred)[0])
        predicted_areas.append(area)
        predicted_depths.append(depth)
    # breakpoint()
    plt.close('all')

    plt.figure(dpi = 600, figsize = np.array([4,3]))
    plt.ylim([0, img_shape])
    plt.xlim([0,1])
    # plt.xlim([0, len(depths)-1])
    # plt.ylim([0, 10])
    sns.set(style = 'ticks')
    plt.title(pv)
    plt.plot((np.arange(len(predicted_depths)))/(len(predicted_depths) - 1), predicted_depths,  alpha = 0.5, linewidth = 2, label = 'Predicted Measurement')

    plt.plot((np.arange(len(depths)))/(len(depths) - 1), depths, 'k-', alpha = 0.5, linewidth = 2, label = 'Experimental Measurement')
    plt.xlabel('Fraction of Melt Path')
    plt.ylabel(r'Depth [$\mu m$]')
    # sns.reset_orig()s
    plt.legend()
    frame_tick()
    plt.savefig(os.path.join(plotting_directory, f'{pv}_depth.png'), bbox_inches = 'tight')
    plt.clf()
    plt.close('all')

def plot_thermal_image_contour_video(pv_thermal_image, pv, information, model, contour_gt = None, img_shape = None, contour_pred = None, plotting_directory = None):
    exposure = information['exposure']
    image_type = information['image_type']
    sequence_type = information['sequence_type']
    sequence_length = information['sequence_length']
    repeat = information['repeat']
    power = float(pv.split('_')[0])
    velocity = float(pv.split('_')[1])


    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi = 150, facecolor='white')
    os.makedirs(os.path.join(plotting_directory, 'animation_with_contour'), exist_ok=True)

    # Function to update the figure for each frame
    for index in tqdm(range(len(pv_thermal_image[pv]))):

        if len(pv_thermal_image[pv].shape) < 5 and repeat:
            input_image = np.repeat(pv_thermal_image[pv][index][None], 3, axis = 0)[None]
            plotting_image = pv_thermal_image[pv][index]
        elif len(pv_thermal_image[pv].shape) < 5:
            input_image = pv_thermal_image[pv][index][None][None]
            plotting_image = pv_thermal_image[pv][index]
        else:
            input_image = pv_thermal_image[pv][index][None]
            plotting_image = pv_thermal_image[pv][index][0][0]
        prediction = model(torch.tensor(input_image).float().to(device))
        # img_shape = int(np.round(np.sqrt(prediction.shape[-1])))
        if img_shape is None:
            if np.log2(prediction.shape[-1]) % 2 != 0:
                img_shape = int(np.round(np.sqrt(prediction.shape[-1]/2)))
            else:
                img_shape = int(np.round(np.sqrt(prediction.shape[-1])))
        contour_pred = prediction.reshape(prediction.shape[0], 2, img_shape,img_shape)[0].argmax(dim = 0).cpu().detach().numpy()
        # plt.clf()
        axs[0].clear()
        axs[1].clear()
        # breakpoint()
        if plotting_image.max() > 100:
             axs[0].imshow(plotting_image, cmap='inferno')#, vmin=0, vmax=100)
        else:
            axs[0].imshow(plotting_image, cmap='inferno', vmin=0, vmax=0.5)
        axs[0].set_title(fr'{exposure} $\mu$s, P: {power} W, V: {velocity} m/s, {image_type}, {sequence_type}, {sequence_length} frames, ' + '{:03d}'.format(index), fontsize = 6)
        axs[0].axis('off')
        axs[1].set_title("Contour")
        tick_width = 1.5
        contours_gt =  measure.find_contours(contour_gt[index, :], 0.5)
        if contour_pred is not None:
            contours_pred = measure.find_contours(contour_pred, 0.5)
            for contour in contours_pred:
                    axs[1].plot(contour[:, 1]/img_shape, -contour[:, 0]/img_shape, linewidth=2, label = 'Prediction')
        for contour in contours_gt:
                axs[1].plot(contour[:, 1]/img_shape, -contour[:, 0]/img_shape, linewidth=2, label = 'Ground Truth')
        axs[1].set_aspect('equal', 'box')
        axs[1].set_xlabel(r'$X/X_{max}$')
        axs[1].set_ylabel(r'$Y/Y_{max}$')
        axs[1].set_xlim(-0.65, 1.5)
        axs[1].set_ylim(-1, 0)
        axs[1].tick_params(direction = 'in', 
                        width = tick_width)
        for axis in ['top','bottom','left','right']:
            axs[1].spines[axis].set_linewidth(2)
        axs[1].legend(frameon = False)
        plt.savefig(os.path.join(plotting_directory, 'animation_with_contour/animation_with_contour{:03d}.png'.format(index)))
    plt.close('all')

def plot_mean_thermal_images(dataset, highlight_time = None, highlight_pv = None, plot_folder = None, show = False, run_path = None):
    pv_thermal_image = dataset.get_pv_thermal_image()
    plt.figure(dpi = 150, figsize = np.array([4, 3])*1.3)
    cmap = plt.cm.jet
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1300))
    # colors = plt.cm.jet(np.linspace(0,1,len(pv_thermal_image.keys())))
    for key in pv_thermal_image.keys():
        alpha = 1
        if highlight_pv is not None:
            alpha = 0.5
            if highlight_pv == key:
                alpha = 1
        p, v = key.split('_')
        density = float(p)/float(v)
        plt.plot(np.linspace(0, 1, len(pv_thermal_image[key])), np.mean(pv_thermal_image[key], axis=(1,2)), color = cmap(density/1300), alpha = alpha)
        if highlight_time is not None and highlight_pv == key:
            plt.scatter(highlight_time/len(pv_thermal_image[key]), np.mean(pv_thermal_image[key], axis=(1,2))[highlight_time], color = cmap(density/1300), alpha = 1)
        plt.xlabel('Fraction of Sequence')
        plt.ylabel('Mean Normalized Temperature')
        plt.title('Energy Density of Thermal Image')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1300))
    plt.colorbar(sm, ax = plt.gca(), label = 'Energy Density (J/mm)')
    frame_tick()
    if show:
        plt.show()
    else:
        plt.tight_layout()
        savefig(os.path.join(plot_folder, 'mean_thermal_images.png'))
        plt.clf()

def plot_mean_thermal_images_video(dataset, information, highlight_time = None, highlight_pv = None, plot_folder = None, show = False, plotting_directory = None):
    exposure = information['exposure']
    image_type = information['image_type']
    sequence_type = information['sequence_type']
    sequence_length = information['sequence_length']
    fig = plt.figure(dpi = 150, figsize = np.array([4, 3])*1.3)
    cmap = plt.cm.jet
    pv_thermal_image = dataset.get_pv_thermal_image()
    for key in pv_thermal_image.keys():
        alpha = 1
        if highlight_pv is not None:
            alpha = 0.5
            if highlight_pv == key:
                alpha = 1
        p, v = key.split('_')
        density = float(p)/float(v)

        if len(pv_thermal_image[key].shape) > 4:
            plotting_sequence = pv_thermal_image[key][:,0, 0]
        else: 
            plotting_sequence = pv_thermal_image[key]

        if len( np.mean(plotting_sequence, axis=(1,2))) < 2:
             plt.scatter(np.linspace(0, 1, len(plotting_sequence)), np.mean(plotting_sequence, axis=(1,2)), color = cmap(density/1300), alpha = alpha)
        else:
            plt.plot(np.linspace(0, 1, len(plotting_sequence)), np.mean(plotting_sequence, axis=(1,2)), color = cmap(density/1300), alpha = alpha)
        plt.xlabel('Fraction of Sequence')
        plt.ylabel('Mean Normalized Temperature')
        plt.title('Energy Density of Thermal Image')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1300))
    plt.colorbar(sm, ax = plt.gca(), label = 'Energy Density (J/mm)')
    if len(pv_thermal_image[highlight_pv].shape) > 4:
        highlighted_sequence = pv_thermal_image[highlight_pv][:, 0, 0]
    else:
         highlighted_sequence = pv_thermal_image[highlight_pv]
    scat = plt.scatter(0/len(highlighted_sequence), np.mean(highlighted_sequence, axis=(1,2))[0], color = cmap(density/1300), alpha = 1)

    os.makedirs(os.path.join(plotting_directory, 'animation_density'), exist_ok=True)
    frame_tick()
    def update(index):
        scat.set_offsets((index/len(highlighted_sequence), np.mean(highlighted_sequence, axis=(1,2))[index]))
        plt.title('Energy Density of Thermal Image' + ', Frame: {:03d}'.format(index), fontsize = 8)
        plt.savefig(os.path.join(plotting_directory, 'animation_density/animation_{:03d}.png'.format(index)))
        return scat

    ani = FuncAnimation(fig, update, frames=len(highlighted_sequence))
    # ani = FuncAnimation(fig, update, frames=2)

    plot_folder = f'result_new_plots/thermal_images/{highlight_pv}/{image_type}/{sequence_type}/{sequence_length}/{exposure}'
    os.makedirs(plot_folder, exist_ok=True)
    # To save the animation
    # ani.save(os.path.join(plot_folder, 'animation.mp4'), codec = 'libx264')
    ani.save(os.path.join(plot_folder, 'animation_density.gif'), writer="imagemagick", fps=10)
    if plotting_directory is not None:
        ani.save(os.path.join(plotting_directory, 'animation_density.gif'), writer="imagemagick", fps=10)
    plt.clf()

def plot_results(dataset, model, model_type, partition, debug_plot = False, plot = False, show = False, run_path = None, folder = 'segment_results', create_videos = False, img_shape = None, sweep_id = None):
    """
    Analyzes and visualizes prediction results from a given model, and performs dimensionality reduction.

    This function processes items from datasets, applies a model for predictions, visualizes these predictions, and performs dimensionality reduction (TSNE and PCA) on the results. 
    It supports multiple visualization techniques and handles the creation of directories for storing these visualizations.

    Parameters:
    - items_info: List of dictionaries containing information about each item.
    - items_pv: Dictionary mapping 'pv' values to corresponding items in the dataset.
    - items_info_pv: Dictionary containing additional information for items in items_pv.
    - model: The predictive model to be used for generating outputs from sample images.
    - model_type: String indicating the type of the model for specific processing (e.g., 'DPT').
    - debug_plot: Boolean flag indicating whether to display debug plots. Defaults to False.
    - plot: Boolean flag indicating whether to create and save plots. Defaults to False.
    - show: Boolean flag indicating whether to display the plot. Defaults to False.
    - run_path: Optional path for saving results and plots. If None, defaults to 'unknown_result'.

    Returns:
    - tuple: Contains two dictionaries, `pv_metrics` with various computed metrics, and `pv_listed_metrics` with listed metrics for each 'pv' combination.

    The function creates and saves plots in specified directories, and handles the creation of these directories if they do not exist.
    """
    items_info_pv, items_pv, items_info, items = dataset.process_parameter_mapping()

    information = {
        'exposure': dataset.exposure,
        'image_type': dataset.image_type,
        'sequence_type': dataset.sequence_type,
        'sequence_length': dataset.sequence_length,
        'repeat': dataset.repeat

        }

    plotting_directory = create_results_directory(run_path, subfolder = 'folder/plot_results', sweep_id = sweep_id)
    dim_reduction_directory = create_results_directory(run_path, subfolder = 'folder/dim_reduction', sweep_id = sweep_id)
    os.makedirs(plotting_directory, exist_ok = True)
    os.makedirs(dim_reduction_directory, exist_ok = True)
    pv_metrics = {}
    pv_listed_metrics = {}

    pv_combinations = np.unique([a[0]['pv'] for a in items_info])
    pv_input_list, pv_list, depth_list = [], [], []
    embedding_list, pred_depth_list, power_list, velocity_list, time_list = [], [], [], [], []
    metrics_list = ['ious', 'mean_iou', 'area_maes', 'areas_gt', 'areas_pred', 'depths_gt', 'depths_pred', 'depths_mae', 'embedding', 'inputs', 'outputs', 'times', 'powers', 'velocities']
    for metric in metrics_list:
        pv_metrics[metric] = {}

    for pv_combo in tqdm(pv_combinations):

        all_ious, all_area_maes, all_areas_gt, all_areas_pred, all_depth_target, all_depth_pred, all_depth_maes = [], [], [], [], [], [], []
        all_inputs, all_outputs, embeddings, powers, velocities, times = [], [], [], [], [], []

        for i in range(len(items_pv[pv_combo])):
            frame_value = items_info_pv[pv_combo][i][0]['frame_number']
            sample_image = torch.tensor(items_pv[pv_combo][i][0]).float()[None].to(device)
            target = torch.tensor(items_pv[pv_combo][i][1]).float().to(device)
            output = model((sample_image.float()))
            # img_shape = int(np.round(np.sqrt(output.shape[-1])))
            if img_shape is None:
                if np.log2(output.shape[-1]) % 2 != 0:
                    img_shape = int(np.round(np.sqrt(output.shape[-1]/2)))
                else:
                    img_shape = int(np.round(np.sqrt(output.shape[-1])))

            if not dataset.sdf:
                binary_img_shape = img_shape 
            else:
                binary_img_shape = 256


            if not dataset.sdf:
                pred_img = output[0].argmax(dim = 0).cpu().detach().numpy()
                target_img = target[0][0].cpu().detach().numpy()
            else:
                pred_img = sdf_to_binary(output[0].reshape(64,64).cpu().detach().numpy(), binary_img_shape)
                target_img = sdf_to_binary(target.reshape(64,64).cpu().detach().numpy(), binary_img_shape)


            if plot:
                fig, axs = plt.subplots(1,4, dpi = 150)
                iou = compute_iou(target_img, pred_img,plot = True, ax = axs[3])
            else:
                iou = compute_iou(target_img, pred_img,plot = False)

            target_depth = np.where(target_img)[0].max()
            
            if model_type == 'DPT':
                embedding = model.dpt.dpt(sample_image.float()).last_hidden_state[0, -1].cpu().detach().numpy()
            if model_type == 'UNet':
                 embedding = model.embed(sample_image.float()).cpu().detach().numpy()
            if model_type == 'ViT':
                embedding = model.vit(model.resize(sample_image.float())).cpu().detach().numpy()
            elif model_type == 'temp_transformer':
                if isinstance(model, torch.nn.DataParallel):
                    embedding = model.module.embed(sample_image.float()).cpu().detach().numpy()[0, -1]
                else: 
                    embedding = model.embed(sample_image.float()).cpu().detach().numpy()[0, -1]
            pred_depth = np.where(pred_img)[0].max()

            conversion_x = IMAGE_EXTENT_X/binary_img_shape
            conversion_y = IMAGE_EXTENT_Y/binary_img_shape
            all_depth_target.append(target_depth*conversion_y)
            all_depth_pred.append(pred_depth*conversion_y)
            all_depth_maes.append(conversion_y*np.abs(target_depth - pred_depth))


            if debug_plot:
                plt.imshow(target_img)
                plt.colorbar()
                plt.title(f'Target Image, Area = {target_img.sum()}')
                plt.imshow(pred_img)
                plt.colorbar()
                plt.title(f'Prediction Image, Area = {pred_img.sum()}')
                plt.show()
                plt.imshow(np.abs(target_img - pred_img))
                plt.title(f'Absolute Difference, MAE = {np.abs(target_img.sum() - pred_img.sum())}, MAPE = {(np.abs(target_img.sum() - pred_img.sum()))/target_img.sum()}')
                plt.tight_layout()
                plt.show()

            all_area_maes.append((np.abs(target_img.sum() - pred_img.sum()))/target_img.sum())
            
            all_areas_gt.append(conversion_x*conversion_y*target_img.sum())
            all_areas_pred.append(conversion_x*conversion_y*pred_img.sum())
            all_ious.append(iou)
            if 'power' in pv_combo:
                power = pv_combo.split('power')[-1].split('_')[0]
                velocity = pv_combo.split('velocity')[-1].split('_')[0]
            else:
                power = pv_combo.split('_')[0]
                velocity = pv_combo.split('_')[1]

            if debug_plot:
                plt.imshow(sample_image[0][0].cpu().detach().numpy(), cmap = 'inferno')
                plt.colorbar()
                plt.show()
            if plot:
                plot_data(sample_image = sample_image, prediction = output, target = target, frame_value = frame_value, 
                          power = power, velocity = velocity, plotting_directory = plotting_directory, 
                          title = f'[Frame {frame_value}] -> [Frame {frame_value + len(items_pv[pv_combo][i][0])}]', 
                          show = show, fig = fig, axs = axs)
        
            powers.append(power)
            velocities.append(velocity)
            times.append(frame_value)
            embeddings.append(np.squeeze(embedding))

        if plot:
            gif_command = f'convert {plotting_directory}/sample_{pv_combo}_*.png {plotting_directory}/sample_{pv_combo}.gif'
            print(gif_command)
            os.system(gif_command)
        pv_metrics['ious'][pv_combo] = all_ious
        pv_metrics['mean_iou'][pv_combo] = np.mean(all_ious)
        pv_metrics['area_maes'][pv_combo] = all_area_maes
        pv_metrics['areas_gt'][pv_combo] = all_areas_gt
        pv_metrics['areas_pred'][pv_combo] = all_areas_pred
        pv_metrics['depths_gt'][pv_combo] = all_depth_target
        pv_metrics['depths_pred'][pv_combo] = all_depth_pred
        pv_metrics['depths_mae'][pv_combo] = all_depth_maes
        pv_metrics['embedding'][pv_combo] = embeddings
        pv_metrics['inputs'][pv_combo] = all_inputs
        pv_metrics['outputs'][pv_combo] = all_outputs
        pv_metrics['times'][pv_combo] = times
        pv_metrics['powers'][pv_combo] = powers
        pv_metrics['velocities'][pv_combo] = velocities
        pv_input_list.extend(all_inputs)
        pv_list.extend(all_outputs)
        depth_list.extend(all_depth_target)
        embedding_list.extend(embeddings)
        pred_depth_list.extend(all_depth_pred)
        power_list.extend(powers)
        velocity_list.extend(velocities)
        time_list.extend(times)
        plt.close('all')
    plt.clf()

    plt.figure(dpi = 150, figsize= np.array([4,3])*1.3)
    plt.xlabel('Ground Truth Depth')
    plt.ylabel('Predicted Depth')
    plt.scatter(depth_list, pred_depth_list, c= np.array(power_list, dtype = float)/np.array(velocity_list, dtype = float), edgecolors = 'k')
    plt.plot([0, 400], [0, 400], 'k--')
    plt.xlim([0, 300])
    plt.ylim([0, 300])
    frame_tick()
    plt.title(f'{partition} Partition, Sequence Rollout Results')
    clb = plt.colorbar()
    clb.ax.set_title("Energy Density [J/mm]")
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'energy_density_depth_{partition}.png'))
    plt.clf()
    pv_thermal_image = dataset.get_pv_thermal_image()
    pv_contours = dataset.get_pv_contours()
  
    if create_videos:
        print('Plotting contour depth')
        plot_contour_depth_area(pv_metrics = pv_metrics,
                                    highlight_pv = list(pv_metrics['embedding'].keys())[0],
                                    plot = 'depth',
                                    plotting_directory = plotting_directory)


        print('Plotting thermal images video with contour')
        plot_thermal_image_contour_video(pv_thermal_image = pv_thermal_image, 
                                            pv = list(pv_metrics['embedding'].keys())[0], 
                                            information = information, 
                                            model = model, 
                                            img_shape = img_shape,
                                            contour_gt = pv_contours[list(pv_metrics['embedding'].keys())[0]], 
                                            plotting_directory = plotting_directory)
        
        print('Plotting average thermal images video')
        plot_mean_thermal_images_video(dataset = dataset,
                                        information = information,
                                        highlight_time = None,
                                        highlight_pv = list(pv_metrics['embedding'].keys())[0],
                                        show = show, 
                                        plotting_directory = plotting_directory)

        print('Plotting contour area')
        plot_contour_depth_area(pv_metrics = pv_metrics,
                                highlight_pv = list(pv_metrics['embedding'].keys())[0],
                                plot = 'area',
                                plotting_directory = plotting_directory)
                
    plt.close('all')
    

    feature_dim_reduction = np.array(embedding_list).reshape(len(embedding_list), -1)
    dim_reduction = True

    if dim_reduction:
        for reduction_model in ['TSNE', 'PCA']:

            if reduction_model == 'TSNE':
                perplexity = np.min([40, len(feature_dim_reduction)//3])
                tsne = TSNE(n_components=2,  perplexity=perplexity, n_iter=300)
                dim_reduction = tsne.fit_transform(feature_dim_reduction)
                text = 'TSNE Perplexity: {}'.format(40)

            elif reduction_model == 'PCA':
                pca = PCA(n_components=2)
                pca.fit(feature_dim_reduction)
                dim_reduction = pca.transform(feature_dim_reduction)  
                text = ''#PCA Variance Explained: {}'.format(pca.explained_variance_ratio_.sum())
            cbounds = {
                'E_Density': [0, 1000],
                'Depth': [0, 450],
                'Depth Error': [0, 30],
                'Normalized': [0, 1]
            }
            plot_dimensionality_reduction(results = dim_reduction, 
                                        labels = np.array(power_list, dtype = 'int')/np.array(velocity_list, dtype = 'float'),
                                        title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}', 
                                        dim_reduction_directory = dim_reduction_directory,
                                        text = text, 
                                        cbar_title = 'Energy Density [J/mm]', 
                                     
                                        figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_energy_density')

            plot_dimensionality_reduction(results = dim_reduction,
                                            labels = np.array(depth_list, dtype = 'float'),
                                            title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}', 
                                            dim_reduction_directory = dim_reduction_directory,
                                            text = text, 
                                            cbar_title = 'Ground Truth Depth [$\mu m$]', 
                                           
                                            figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_gt_depth')
            plot_dimensionality_reduction(results = dim_reduction,
                                            labels = np.array(pred_depth_list, dtype = 'float'),
                                            title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}', 
                                            dim_reduction_directory = dim_reduction_directory,
                                            text = text, 
                                         
                                            cbar_title = 'Predicted Depth [$\mu m$]', 
                                            figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_pred_depth')
            plot_dimensionality_reduction(results = dim_reduction,
                                            labels = np.abs(np.array(pred_depth_list, dtype = 'float') - np.array(depth_list, dtype = 'float')),
                                            title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}', 
                                            dim_reduction_directory = dim_reduction_directory,
                                            text = text, 
                                            cbar_title = 'Depth Error [$\mu m$]', 
                                            cbounds = cbounds['Depth Error'],
                                            figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_depth_error')
            plot_dimensionality_reduction(results = dim_reduction,
                                            labels = np.abs(np.array(pred_depth_list, dtype = 'float') - np.array(depth_list, dtype = 'float'))/np.array(depth_list, dtype = 'float'),
                                            title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}', 
                                            dim_reduction_directory = dim_reduction_directory,
                                            text = text, 
                                            cbar_title = 'Normalized Depth Error',
                                            cbounds = cbounds['Normalized'],
                                            figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_depth_percent_error')
            if reduction_model == 'PCA' and create_videos:
                plot_dimensionality_reduction_video(pc_model = pca,
                                                    results = dim_reduction,
                                                    labels = np.array(depth_list, dtype = 'float'),
                                                    pv_embeddings = pv_metrics['embedding'],
                                                    highlighted_pv =list(pv_metrics['embedding'].keys())[0],
                                                    title = f'{reduction_model} {partition.capitalize()} Set, model type {model_type}',
                                                    dim_reduction_directory = plotting_directory,
                                                    text = text,
                                                    cbar_title = 'Experimentally Measured Depth',
                                                    figure_fname = f'rollout_{partition}_{reduction_model}_{model_type}_gt_depth',
                                                    information = information)
                
    pv_listed_metrics = {
        'inputs': pv_input_list,
        'outputs': pv_list,
        'depths': depth_list,
        'embeddings': embedding_list,
        'times': time_list,
        'powers': power_list,
        'velocities': velocity_list
    }

    return pv_metrics, pv_listed_metrics
def resample_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    # Parametric distance along the contour
    distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)  # insert 0 at start

    # Interpolation functions
    interp_x = interp1d(distance, contour[:, 0], kind='linear')
    interp_y = interp1d(distance, contour[:, 1], kind='linear')

    # New parametric distance (100 points)
    new_distance = np.linspace(0, distance[-1], 100)

    # New contour

    new_contour = np.vstack([interp_x(new_distance), interp_y(new_distance)]).T
    return new_contour
def plot_metrics(dataset, partition, model, model_type, debug_plot=False, run_path=None, show=False, create_videos=False, img_shape = None, sweep_id = None):
    """
    Plot the metrics for the given dataset and model.

    Args:
        dataset (Dataset): The dataset to evaluate.
        partition (str): The partition of the dataset ('train', 'val', or 'test').
        model (Model): The model to evaluate.
        model_type (str): The type of the model.
        debug_plot (bool, optional): Whether to enable debug plots. Defaults to False.
        run_path (str, optional): The path to save the results. Defaults to None.
        show (bool, optional): Whether to show the plots. Defaults to False.
        create_videos (bool, optional): Whether to create videos. Defaults to False.

    Returns:
        dict: The error metrics calculated.
    """
    metrics_directory = create_results_directory(run_path, subfolder='metrics', sweep_id = sweep_id)
    plotting_directory = create_results_directory(run_path, subfolder='plot_metrics', sweep_id = sweep_id)
    print('metrics saved here')
    os.makedirs(metrics_directory, exist_ok=True)
    os.makedirs(plotting_directory, exist_ok=True)
    print(metrics_directory)

    pv_metrics, pv_listed_metrics = plot_results(dataset=dataset,
                                                model=model,
                                                partition=partition,
                                                model_type=model_type,
                                                debug_plot=debug_plot,
                                                plot=False,
                                                show=show,
                                                run_path=run_path,
                                                folder='original', create_videos=create_videos, 
                                                img_shape = img_shape)
    print(pv_metrics.keys())

    if model_type != 'temp_transformer':
        dataset.sequence_type = 'segment_mean'
        # dataset.augment = True

    pv_metrics, pv_listed_metrics = plot_results(dataset=dataset,
                                                model=model,
                                                partition=partition,
                                                model_type=model_type,
                                                debug_plot=debug_plot,
                                                plot=False,
                                                show=show,
                                                run_path=run_path,
                                                folder='segment', create_videos=create_videos, 
                                                img_shape = img_shape)
    pv_mean_ious = pv_metrics['mean_iou']
    pv_ious = pv_metrics['ious']
    pv_area_maes = pv_metrics['area_maes']
    pv_areas_gt = pv_metrics['areas_gt']
    pv_areas_pred = pv_metrics['areas_pred']
    pv_depths_gt = pv_metrics['depths_gt']
    pv_depths_pred = pv_metrics['depths_pred']
    pv_depths_mae = pv_metrics['depths_mae']
    with open(os.path.join(metrics_directory, 'pv_ious.json'), 'w') as f:
        json.dump(pv_ious, f)
    with open(os.path.join(metrics_directory, 'pv_mean_ious.json'), 'w') as f:
        json.dump(pv_mean_ious, f)
    with open(os.path.join(metrics_directory, 'pv_area_maes.json'), 'w') as f:
        json.dump(pv_area_maes, f)
    with open(os.path.join(metrics_directory, 'pv_areas_gt.json'), 'w') as f:
        json.dump(pv_areas_gt, f)
    with open(os.path.join(metrics_directory, 'pv_areas_pred.json'), 'w') as f:
        json.dump(pv_areas_pred, f)
    with open(os.path.join(metrics_directory, 'pv_depths_gt.json'), 'w') as f:
        json.dump(pv_depths_gt, f)
    with open(os.path.join(metrics_directory, 'pv_depths_pred.json'), 'w') as f:
        json.dump(pv_depths_pred, f)
    plt.close('all')
    mean_iou = []
    e_densities = []
    std_iou = []
    powers = []
    velocities = []
    area_maes = []
    area_std_maes = []
    area_raw_maes = []
    area_raw_std_maes = []
    all_ious = []
    all_area_maes = []
    all_area_raw_maes = []

    for key in pv_mean_ious.keys():
        p, v = key.split('_')
        powers.append(int(p))
        velocities.append(float(v))
        energy_density = float(p) / (1000 * float(v))

        e_densities.append(energy_density)
        mean_iou.append(pv_mean_ious[key])
        std_iou.append(np.std(pv_ious[key]))
        area_maes.append(np.mean(pv_area_maes[key]))
        area_std_maes.append(np.std(pv_area_maes[key]))

        area_raw_maes.append(np.mean(np.abs(np.array(pv_areas_gt[key]) - np.array(pv_areas_pred[key]))))
        area_raw_std_maes.append(np.std(np.abs(np.array(pv_areas_gt[key]) - np.array(pv_areas_pred[key]))))
        all_ious.extend(pv_ious[key])
        all_area_maes.extend(pv_area_maes[key])
        all_area_raw_maes.extend(np.abs(np.array(pv_areas_gt[key]) - np.array(pv_areas_pred[key])))

    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    plt.scatter(e_densities, mean_iou, edgecolor='k', c=powers)
    plt.errorbar(e_densities, mean_iou, yerr=std_iou, fmt='none', c='k', capsize=3)
    clb = plt.colorbar()
    clb.ax.set_title("Power [W]")
    plt.xlabel(r'Energy Density [$J/{m^3}$]')
    plt.ylabel('Mean IoU')
    plt.ylim([0, 1])
    frame_tick()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'rollout_energy_density_iou_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    plt.close('all')
    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    plt.scatter(e_densities, mean_iou, edgecolor='k', c=velocities)
    plt.errorbar(e_densities, mean_iou, yerr=std_iou, fmt='none', c='k', capsize=3)

    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean IoU')
    plt.ylim([0, 1])
    frame_tick()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'rollout_energy_density_iou_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    plt.close('all')
    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    plt.scatter(e_densities, 100 * np.array(area_maes), edgecolor='k', c=velocities)
    plt.errorbar(e_densities, 100 * np.array(area_maes), yerr=100 * np.array(area_std_maes), fmt='none', c='k', capsize=3)

    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean Absolute Error [%]')
    frame_tick()
    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'rollout_energy_density_area_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    plt.close('all')
    plt.figure(dpi=150, figsize=np.array([4, 3]) * 1.3)
    plt.scatter(e_densities, np.array(area_raw_maes), edgecolor='k', c=velocities)
    plt.errorbar(e_densities, np.array(area_raw_maes), yerr=np.array(area_raw_std_maes), fmt='none', c='k', capsize=3)
    
    clb = plt.colorbar()
    clb.ax.set_title("V [m/s]")
    plt.xlabel(r'Energy Density ($J/{m^3}$)')
    plt.ylabel('Mean Absolute Error [$\mu m^2$]')
    frame_tick()

    title_partition = partition[0].upper() + partition[1:]
    plt.title(f'{title_partition} Set')
    plt.grid()
    plt.tight_layout()
    savefig(os.path.join(plotting_directory, f'rollout_energy_density_area_raw_{partition}.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    error_metrics = {'rollout_iou': np.mean(all_ious), 'rollout_iou_std': np.std(all_ious),
                     'rollout_area_percent_error': np.mean(all_area_maes),
                     'rollout_area_percent_error_std': np.std(all_area_maes),
                     'rollout_area_error': np.mean(all_area_raw_maes),
                     'rollout_area_error_std': np.std(all_area_raw_maes)}
    return error_metrics



import scipy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_melt_pool(image, directory, power = '', velocity = ''):

    plt.figure(dpi=300)
    plt.rcParams['text.usetex'] = False
    # Parameters for plotting
    num_melt_pools = 5
    start_alpha = 0.35
    alpha_increment = (1 - start_alpha) / (num_melt_pools - 1)
    start = 0
    composite_height, composite_width = 64, 2*64 # Assume these are set based on your data

    # Create a blank RGBA image for the composite

    composite_rgba = np.zeros((composite_height, int(3.8 * composite_width), 4))
    composite_meltpool = np.zeros((composite_height, int(3.8 * composite_width)))

    for i in range(num_melt_pools):
        
        temperature_values = image[-1].cpu().detach().numpy()[i+5*i,0,:,:].T
        
        cropped_meltpool = temperature_values*65536/10#[:-1, 50:150]
        current_alpha = 1#start_alpha + i * alpha_increment

        # Normalize temperature values for color mapping
        minimum  = np.min(cropped_meltpool[cropped_meltpool > 0])
        maximum = np.max(cropped_meltpool)
        
        normalized_meltpool = (cropped_meltpool - np.min(cropped_meltpool)) / (np.max(cropped_meltpool) - np.min(cropped_meltpool))
        
        # Convert to RGBA using the colormap
        cmap = plt.get_cmap('inferno')
        # cropped_meltpool[cropped_meltpool < 300] = np.nan
        rgba_meltpool = cmap(normalized_meltpool)

        
        # Adjust alpha channel
        rgba_meltpool[..., 3] = rgba_meltpool[..., 3] * current_alpha
        
        # Update the composite image
        # print(rgba_meltpool.shape, composite_rgba.shape)
        composite_rgba[:, start:start+cropped_meltpool.shape[1], :] = rgba_meltpool

        composite_meltpool[:, start:start+cropped_meltpool.shape[1]] += cropped_meltpool
        # print(start, start+cropped_meltpool.shape[1])
        text_x_position = start + (cropped_meltpool.shape[1] / 2) - 10
        
        # Adjust the y position below the melt pool; you may need to tweak this depending on your figure size
        text_y_position = cropped_meltpool.shape[0] + 5  # Adjust '10' as needed to move the text lower

        # composite_rgba[]

        start += 75  # Adjust based on your layout

    # Plot the composite image

    composite_meltpool[composite_meltpool < 300] = np.nan

    im = plt.imshow(composite_meltpool[:, :450], vmin = 2000, vmax=3500, cmap = 'inferno', extent=[0, 3.8*composite_width*5.6, 0, composite_height*5.6])

    # clb = plt.colorbar(im,fraction=0.046, pad=0.04)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    # plt.axis('off')
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, aspect=20)

    # clb.ax.set_title('$T [K]$')
    savefig(os.path.join(directory, f'test_sample_image_power_{power}_velocity_{velocity}.png'))
    savefig(os.path.join(directory, f'test_sample_image_power_{power}_velocity_{velocity}.svg'))
    plt.axis('on')
    frame_tick()
    plt.grid()
    plt.xlabel('X [ $\mu m$]')
    plt.ylabel('Y [ $\mu m$]')
    plt.title(f'$P$ = {power} W, $V$ = {velocity} m/s')
    savefig(os.path.join(directory, f'test_axis_sample_image_power_{power}_velocity_{velocity}.png'))
    savefig(os.path.join(directory, f'test_axis_sample_image_power_{power}_velocity_{velocity}.svg'))
    plt.clf()
    plt.close('all')

def plot_overlap(contour_gt, contour_pred, directory, power = '', velocity = ''):

    sns.set(style="ticks", color_codes=True)
    plt.figure(dpi = 300, figsize = np.array([4,3])*1.2)
    plt.fill(contour_pred[:, 1], -contour_pred[:, 0], linewidth=2, alpha = 0.2)
    plt.fill(contour_gt[:, 1],  -contour_gt[:, 0],  linewidth=2, alpha = 0.2)
    plt.plot(contour_pred[:, 1], -contour_pred[:, 0], linewidth=2, alpha = 1, label = 'Predicted')
    plt.plot(contour_gt[:, 1], -contour_gt[:, 0],  linewidth=2, alpha = 1, label = 'Ground Truth')
    plt.ylim(-450, 0)
    plt.xlim(0, 450)
    frame_tick()
    plt.ylim(-450, 0)
    plt.xlim(0, 450)

    ax = plt.gca()

    ax.set_aspect(abs((ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])))
    plt.ylabel(r'Depth $[\mu m]$')
    plt.xlabel(r'Width $[\mu m]$')
    plt.legend(frameon = False, loc = 'lower right')
    # plt.show()
    power = int(power)
    plt.title(f'$P$ = {power} W, $V$ = {velocity} m/s')
    savefig(os.path.join(directory, f'test_sample_contour_power_{power}_velocity_{velocity}.png'))
    savefig(os.path.join(directory, f'test_sample_contour_power_{power}_velocity_{velocity}.svg'))
    plt.close('all')

import pandas as pd
def plot_train_test_metrics(model, train_dataloader, test_dataloader, run_path = None, show = False, img_shape = None, sweep_id = '', config = None,
                             plot_publication_figures = False):
        plotting_directory = create_results_directory(run_path, subfolder = 'plot_train_test', sweep_id = sweep_id)
        summary_directory = create_results_directory(run_path, subfolder = 'summary', sweep_id = sweep_id)
        error_metrics = {}
        area_gt = {}
        powers = {}
        velocities = {}
        area_pred = {}
        area_gt = {}
        depth_gt = {}
        depth_pred = {}
        ious = {}
        cross_entropy = {}
        contour_mae = {}
        contour_mse = {}
        hausdorff = {}
       

        plot = False
        criterion  = nn.CrossEntropyLoss()
        # partition = 'Train'
        for partition in ['Test']:#, 'Train', ]:
                if partition == 'Test' or partition ==  'test':
                        dataloader = test_dataloader
                if partition == 'Validation' or partition ==  'validation':
                        dataloader = test_dataloader
                elif partition == 'Train' or partition == 'train':
                        dataloader = train_dataloader
                
                powers[partition] = []
                velocities[partition] = []
                area_gt[partition] = []
                area_pred[partition] = []
                depth_gt[partition] = []
                depth_pred[partition] = []
                ious[partition] = []
                cross_entropy[partition] = []
                contour_mae[partition] = []
                contour_mse[partition] = []
                hausdorff[partition] = []
                plot_cases = 0
                plot_contour_cases = 0
                n_plotted = 0
                for batch_num, (image, target, index) in tqdm(enumerate(dataloader), total = len(dataloader)):
                        # print(image.shape)
                        image = image.to(device)
                        target = target.to(device)
                        # print(image.shape)
                        output = model((image.float()))
                        if len(output.shape) < 4:
                            if img_shape is None:
                                # cross_entropy_output = output.reshape(target.shape[0], 2, -1)
                                if np.log2(image.shape[-1]) % 2 != 0:
                                    img_shape = int(np.round(np.sqrt(output.shape[-1]/2)))
                                else:
                                    img_shape = int(np.round(np.sqrt(output.shape[-1])))
                            if not config['sdf']:
                                output = output.reshape(output.shape[0], 2, img_shape, img_shape)
                                loss = criterion((output.reshape(target.shape[0], 2,-1)).float(), target.long().reshape(target.shape[0], -1))
                            else:
                                output = output.reshape(output.shape[0], 1, img_shape, img_shape)
                                loss = 0# criterion((output.reshape(target.shape[0], 1,-1)).float(), target.long().reshape(target.shape[0], -1))
                        
                        
                        
                        if plot and plot_cases < 3:
                            if config['sdf']:
                                raise NotImplementedError
                            else:
                                plt.clf()
                                fig, axs = plt.subplots(1,3, dpi = 150)
                                # axs.plot(x, y)
                                if len(image.shape) == 5:
                                     # Temporal sequence
                                    axs[0].imshow(image[0][0].cpu().detach().numpy()[0], cmap = 'inferno')
                                else:
                                    axs[0].imshow(image[0][0].cpu().detach().numpy(), cmap = 'inferno')
                              
                                axs[1].imshow(output[0].argmax(dim  = 0).cpu().detach().numpy())
                                axs[2].imshow(target[0][0].cpu().detach().numpy())
                                axs[0].set_title('Input Sample [Frame 0]')
                                axs[1].set_title('Prediction')
                                axs[2].set_title('Ground Truth')
                                axs[0].axis('off')
                                axs[1].axis('off')
                                axs[2].axis('off')
                                # if run_path is not None:
                                plt.tight_layout()

                                savefig(os.path.join(plotting_directory, f'sample_{partition}_{plot_cases}.png'))
                                if show:
                                        plt.show()
                                else:
                                        plt.clf()
                                plot_cases += 1
                        
                        pv_string = dataloader.dataset.__getiteminfo__(index.item())[0]['pv']
                        if 'power' in pv_string and 'velocity' in pv_string:
                                # FLOW-3D dataset
                                power = int(pv_string.split('_')[0].split('power')[1])
                                velocity = int(pv_string.split('_')[1].split('velocity')[1])
                        else:
                            power = float(pv_string.split('_')[0])
                            velocity = float(pv_string.split('_')[1])



                        if plot and plot_contour_cases < 15:
                            if config['sdf']:
                                raise NotImplementedError
                            else:
                                fig.suptitle(f'{partition} Prediction, Power = {power} W, Velocity = {velocity} m/s', y = 0.8)
                                fig.tight_layout()
                                
                                # img_shape = int(np.round(np.sqrt(output.shape[-1])))
                                if img_shape is None:
                                    if np.log2(output.shape[-1]) % 2 != 0:
                                        img_shape = int(np.round(np.sqrt(output.shape[-1]/2)))
                                    else:
                                        img_shape = int(np.round(np.sqrt(output.shape[-1])))
                                contour_pred = measure.find_contours(output[0].argmax(dim = 0).cpu().detach().numpy(), 0.5)
                                contour_gt =  measure.find_contours(target[0][0].cpu().detach().numpy())
                                # Display the image and plot all contours found
                                fig, ax = plt.subplots(dpi = 150, figsize = np.array([4,3])*1.3)

                                for contour in contour_pred:
                                        ax.plot(contour[:, 1], -contour[:, 0], linewidth=2, label = 'Prediction')
                                for contour in contour_gt:
                                        ax.plot(contour[:, 1], -contour[:, 0], linewidth=2, label = 'Ground Truth')
                                plt.ylim(-img_shape,0)
                                plt.xlim(0, img_shape)
                                plt.title(f'{partition} Prediction, Power = {power} W, Velocity = {velocity} m/s')
                                frame_tick()
                                plt.legend()
                                plt.tight_layout()

                                savefig(os.path.join(plotting_directory, f'contour_{partition}_{plot_contour_cases}.png'))
                                if show:
                                        plt.show()
                                else:
                                        plt.clf()
                                plt.close('all')
                                plot_contour_cases += 1

                        powers[partition].append(power)
                        velocities[partition].append(velocity)
                        # img_shape = int(np.round(np.sqrt(output.shape[-1])))
                        # img_shape = int(np.round(np.sqrt(output.shape[-1])))

                        if not config['sdf']:
                            binary_img_shape = img_shape 
                        else:
                            binary_img_shape = 64
                        # if img_shape is None:
                        #     if np.log2(output.shape[-1]) % 2 != 0:
                        #         img_shape = int(np.round(np.sqrt(output.shape[-1]/2)))
                        #     else:
                        #         img_shape = int(np.round(np.sqrt(output.shape[-1])))



                        conversion_x = IMAGE_EXTENT_X/binary_img_shape
                        conversion_y = IMAGE_EXTENT_Y/binary_img_shape
                        if not config['sdf']:
                            binary_pred = output[0].argmax(dim = 0).cpu().detach().numpy()
                            target_image = target[0][0].cpu().detach().numpy()
                        else:
                            binary_pred = sdf_to_binary(output[0].reshape(64,64).cpu().detach().numpy(), binary_img_shape)
                            target_image = sdf_to_binary(target[0].reshape(64,64).cpu().detach().numpy(), binary_img_shape)
                        # breakpoint()
                        try:
                            contour_pred = measure.find_contours(binary_pred, 0.5)[0]*np.array([conversion_y, conversion_x])
                        except IndexError as e:
                            print("No contour found for this prediction")
                            contour_pred = np.array([[0,0]])
                        contour_gt =  measure.find_contours(target_image, 0.5)[0]*np.array([conversion_y, conversion_x])
                        new_pred_contour = resample_contour(contour_pred)
                        new_gt_contour = resample_contour(contour_gt)
                        rmse = np.sqrt(np.mean((new_pred_contour - new_gt_contour)**2))
                        mae = np.mean(np.abs(new_pred_contour - new_gt_contour))
                        contour_mae[partition].append(mae)
                        contour_mse[partition].append(rmse)
                        area_pred[partition].append(conversion_x*conversion_y*np.sum(binary_pred))
                        pred_img = binary_pred
                        target_img = target_image
                        area_gt[partition].append(conversion_x*conversion_y*np.sum(target_img))
                        try:
                            depth_pred[partition].append(conversion_y*np.where(pred_img)[0].max())
                        except ValueError:
                            depth_pred[partition].append(0)
                        depth_gt[partition].append(conversion_y*np.where(target_img)[0].max())
                        ious[partition].append(compute_iou(pred_img, target_img))
                        hausdorff[partition].append(scipy.spatial.distance.directed_hausdorff(new_pred_contour, new_gt_contour))
                        # cross_entropy[partition].append(loss.item())
                        cross_entropy[partition].append(0)
                # breakpoint()
                        # plot_publication_figures = True
                        n_plotted += 1
                        if plot_publication_figures and partition == 'Test':
                            if n_plotted < 30:
                                plot_overlap(contour_gt, contour_pred, plotting_directory, power = power, velocity = velocity)
                                if config['model_type'] == 'temp_transformer':
                                    plot_melt_pool(image, plotting_directory, power = power, velocity = velocity)
                            else:
                                continue
                area_pred[partition] = np.array(area_pred[partition])
                area_gt[partition] = np.array(area_gt[partition])
                depth_pred[partition] = np.array(depth_pred[partition])
                depth_gt[partition] = np.array(depth_gt[partition])
                contour_mae[partition] = np.array(contour_mae[partition])
                contour_mse[partition] = np.array(contour_mse[partition])
                hausdorff[partition] = np.array(hausdorff[partition])
                area_error = np.mean(np.abs(area_pred[partition] - area_gt[partition]))
                area_percent_error = np.mean(np.abs(area_pred[partition] - area_gt[partition])/area_gt[partition])
                area_std_error = np.std(np.abs(area_pred[partition] - area_gt[partition]))
                area_std_percent_error = np.std(np.abs(area_pred[partition] - area_gt[partition])/area_gt[partition])
                depth_error = np.mean(np.abs(depth_pred[partition] - depth_gt[partition]))
                depth_percent_error = np.mean(np.abs(depth_pred[partition] - depth_gt[partition])/depth_gt[partition])
                depth_std_error = np.std(np.abs(depth_pred[partition] - depth_gt[partition]))
                depth_std_percent_error = np.std(np.abs(depth_pred[partition] - depth_gt[partition])/depth_gt[partition])

                iou = np.mean(ious[partition])
                iou_std = np.std(ious[partition])
                cross_entropy_mean = np.mean(cross_entropy[partition])
                cross_entropy_std = np.std(cross_entropy[partition])
                r2_area = sklearn.metrics.r2_score(area_gt[partition], area_pred[partition])
                r2_depth = sklearn.metrics.r2_score(depth_gt[partition], depth_pred[partition])
                contour_mae_mean = np.mean(contour_mae[partition])
                contour_mae_std = np.std(contour_mae[partition])
                contour_mse_mean = np.mean(contour_mse[partition])
                contour_mse_std = np.std(contour_mse[partition])
                hausdorff_mean = np.mean(hausdorff[partition])
                hausdorff_std = np.std(hausdorff[partition])
                error_metrics[partition] = {
                    'area_error': area_error,
                    'area_percent_error': area_percent_error,
                    'area_std_error': area_std_error,
                    'area_std_percent_error': area_std_percent_error,
                    'depth_error': depth_error,
                    'depth_percent_error': depth_percent_error,
                    'depth_std_error': depth_std_error,
                    'depth_std_percent_error': depth_std_percent_error,
                    'iou': iou,
                    'iou_std': iou_std,
                    'cross_entropy': cross_entropy_mean,
                    'cross_entropy_std': cross_entropy_std,
                    'r2_area': r2_area,
                    'r2_depth': r2_depth,
                    'contour_mae': contour_mae_mean,
                    'contour_mae_std': contour_mae_std,
                    'contour_rmse': contour_mse_mean,
                    'contour_rmse_std': contour_mse_std,
                    'hausdorff': hausdorff_mean,
                    'hausdorff_std': hausdorff_std
                }
        with open(os.path.join(summary_directory, 'error_metrics.json'), 'w') as f:
                json.dump(error_metrics, f)
       
        for partition in area_gt.keys():
            publication_style = True

            plt.figure(dpi = 600, figsize = np.array([4,4])*1.15)
        
           
            errorbar = False
            for errorbar in [True, False]:
                if publication_style:

                    if errorbar:
                        plt.figure(dpi = 600, figsize = np.array([4,4])*1.15)

                        data = pd.DataFrame({
                                'Area_GT': area_gt[partition]/1000,
                                'Area_Pred': area_pred[partition]/1000,
                                'power': powers[partition],
                                'velocity': velocities[partition],
                                'energy_density': np.array(powers[partition])/(np.array(velocities[partition])*1000)})
                        # breakpoint()
                        # Calculate mean and standard deviation for each unique Depth_GT
                        stats = data.groupby('energy_density').agg(['mean', 'std']).reset_index()

                        # Extract data for plotting
                        depths = stats['Area_GT', 'mean']
                        means = stats['Area_Pred', 'mean']
                        std_devs = stats['Area_Pred', 'std']

                        im = plt.errorbar(depths, means, yerr=std_devs, c = 'k', capsize=3.5,  fmt = 'none', ecolor='k')
                        im = plt.scatter(depths, means, edgecolors = 'k', c = np.array(stats['energy_density']), vmin = 0, vmax = 1.5, s = 50)
                    else:
                        plt.figure(dpi = 600, figsize = np.array([4,4])*1.15)

                        im = plt.scatter(area_gt[partition]/1000, area_pred[partition]/1000, edgecolors = 'k', c = np.array(powers[partition])/(np.array(velocities[partition])*1000), s = 50, vmin = 0, vmax = 1.5, alpha = 0.05)
                    # im = plt.scatter(area_gt[partition]/1000, area_pred[partition]/1000, edgecolors = 'k', c = np.array(powers[partition])/(np.array(velocities[partition])*1000), s = 50, vmin = 0, vmax = 1.5)
                    clb = plt.colorbar(im,fraction=0.046, pad=0.04, label = 'Energy Density [J/mm]')
                    error = np.array(area_gt[partition]) - np.array(area_pred[partition])
                    plt.plot([0, 70], [0, 70], 'k-')
                    plt.plot([0, 70], [0-6.5,70-6.5], 'k--')
                    plt.plot([0, 70], [0+6.5, 70+6.5], 'k--')
                    sns.set(style = 'ticks')

                    plt.xlabel(r'Ground Truth Area $[\times 10^{3}  \mu m^2]$')
                    plt.ylabel(r'Predicted Area $[\times 10^{3} \mu  m^2]$')
                    # r2 = sklearn.metrics.r2_score(area_gt[partition], area_pred[partition])
                    # plt.text(2500,60700, rf'{partition} $R^2$ = ' + '{:.2f}'.format(r2))
                    # plt.axis('equal')
                    ax = plt.gca()
                    # ax.set_aspect('equal', adjustable='box')
                    plt.axis('scaled')
                    plt.xlim(0, 65)
                    plt.ylim(0, 65)
                    frame_tick()

                    plt.tight_layout()
                    if errorbar:
                        savefig(os.path.join(plotting_directory, f'error_bar_area_{partition}.png'))
                        savefig(os.path.join(plotting_directory, f'error_bar_area_{partition}.svg'))
                    else:
                        savefig(os.path.join(plotting_directory, f'area_{partition}.png'))
                        savefig(os.path.join(plotting_directory, f'area_{partition}.svg'))
                    plt.clf()

         
            plt.close('all')
            plt.figure(dpi = 150, figsize = np.array([4,4])*1.3)
            plt.scatter(area_gt[partition], ious[partition], edgecolors = 'k')
            sns.set(style = 'ticks')

            plt.xlabel('Ground Truth Area')
            plt.ylabel('IoU')
            plt.title(f'{partition} Set IoU')
            # plt.xlim(0, 1000)
            plt.ylim(0, 1)
            frame_tick()
            plt.tight_layout()
            savefig(os.path.join(plotting_directory, f'area_iou_{partition}.png'))
            savefig(os.path.join(plotting_directory, f'area_iou_{partition}.svg'))


            if show:
                    plt.show()
            else:
                    plt.clf()

            plt.close('all')
            mae = sklearn.metrics.mean_absolute_error(area_gt[partition], area_pred[partition])

            mape= sklearn.metrics.mean_absolute_percentage_error(area_gt[partition], area_pred[partition])
           
        for partition in depth_gt.keys():
                # errorbar = True
                plt.figure(dpi = 600, figsize = np.array([4,4])*1.15)
                for errorbar in [True, False]:
                    if errorbar:
                        data = pd.DataFrame({
                            'Depth_GT': depth_gt[partition],
                            'Depth_Pred': depth_pred[partition],
                            'power': powers[partition],
                            'velocity': velocities[partition],
                            'energy_density': np.array(powers[partition])/(np.array(velocities[partition])*1000)})
                       
                        # Calculate mean and standard deviation for each unique Depth_GT
                        stats = data.groupby('energy_density').agg(['mean', 'std']).reset_index()

                        # Extract data for plotting
                        depths = stats['Depth_GT', 'mean']
                        means = stats['Depth_Pred', 'mean']
                        std_devs = stats['Depth_Pred', 'std']

                        im = plt.errorbar(depths, means, yerr=std_devs, c = 'k', capsize=3,  fmt = 'none', ecolor='k')
                        #plt.errorbar(e_densities, mean_iou, yerr=std_iou, fmt='none', c='k', capsize=3)
                        # im = plt.errorbar(depths, means, yerr=std_devs, c = np.array(powers[partition])/(np.array(velocities[partition])*1000), vmin = 0, vmax = 1.5, s = 50, fmt='o', capsize=5, capthick=2, ecolor='red', marker='s')
                        im = plt.scatter(depths, means, edgecolors = 'k', c = np.array(stats['energy_density']), vmin = 0, vmax = 1.5, s = 50)
                    else:
                        im = plt.scatter(depth_gt[partition], depth_pred[partition], edgecolors = 'k', c = np.array(powers[partition])/(np.array(velocities[partition])*1000), vmin = 0, vmax = 1.5, s = 50, alpha = 0.05)
                    # im = plt.scatter(depth_gt[partition], depth_pred[partition], edgecolors = 'k', c = np.array(powers[partition])/(np.array(velocities[partition])*1000), vmin = 0, vmax = 1.5, s = 50)
                    clb = plt.colorbar(im,fraction=0.046, pad=0.04, label = 'Energy Density [J/mm]')
                    error = np.array(depth_gt[partition]) - np.array(depth_pred[partition])
                    
                    plt.plot([0, 600], [0, 600], 'k-')
                    plt.plot([0, 600], [0-45,600-45], 'k--')
                    plt.plot([0, 600], [0+45, 600+45], 'k--')
                    # breakpoint()
                    plt.xlabel('Ground Truth Depth $[\mu m]$')
                    plt.ylabel('Predicted Depth $[\mu m]$')
                    # plt.title(f'{partition} Set Depth ')

                    r2 = sklearn.metrics.r2_score(depth_gt[partition], depth_pred[partition])
                    if not publication_style:
                        plt.text(conversion_y*5,420, rf'{partition} $R^2$ = ' + '{:.2f}'.format(r2))
                    # plt.axis('equal')
                    ax = plt.gca()
                    # ax.set_aspect('equal', adjustable='box')
                    plt.axis('scaled')
                    plt.xlim(0, 450)
                    plt.ylim(0, 450)
                    frame_tick()
                    plt.tight_layout()
                    if errorbar:
                        savefig(os.path.join(plotting_directory, f'error_bar_depth_{partition}.png'))
                        savefig(os.path.join(plotting_directory, f'error_bar_depth_{partition}.svg'))
                    else:
                        savefig(os.path.join(plotting_directory, f'depth_{partition}.png'))
                        savefig(os.path.join(plotting_directory, f'depth_{partition}.svg'))
                    plt.clf()

                if show:
                        plt.show()
                else:
                        plt.clf()
                plt.close('all')
                plt.figure(dpi = 150, figsize = np.array([4,4])*1.3)
                plt.scatter(depth_gt[partition], ious[partition], edgecolors = 'k')
                plt.xlabel('Ground Truth Depth')
                plt.ylabel('IoU')
                plt.title(f'{partition} Set IoU')
                # plt.xlim(0, 1000)
                plt.ylim(0, 1)
                frame_tick()
                plt.tight_layout()
                savefig(os.path.join(plotting_directory, f'depth_iou_{partition}.png'))

                if show:
                        plt.show()
                else:
                        plt.clf()

                plt.close('all')
                mae = sklearn.metrics.mean_absolute_error(depth_gt[partition], depth_pred[partition])
    
                mape= sklearn.metrics.mean_absolute_percentage_error(depth_gt[partition], depth_pred[partition])
               