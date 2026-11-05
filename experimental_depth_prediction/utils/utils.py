import json
import os
import torch
import skimage
from experimental_depth_prediction.datasets.contour_area_image import ContourAreaDataset
from experimental_depth_prediction.datasets.flow3d_dataset import Flow3DDataset
from experimental_depth_prediction.datasets.eagar_tsai_dataset import ETDataset


from experimental_depth_prediction.models.frames_to_area_ViT_embedded import VitFineTune, VitFromScratch
from experimental_depth_prediction.models.frames_to_area_CNN_simple import CNN
from experimental_depth_prediction.models.Unet import UNet
from experimental_depth_prediction.models.Unet import UNet_concat
from experimental_depth_prediction.models.ResNet import ResNet

from collections import OrderedDict
from experimental_depth_prediction.utils.plotting_utils import frame_tick
import argparse
from torch.utils.data import DataLoader
import wandb
from experimental_depth_prediction.models.frames_to_area_CNN_LSTM import CNNLSTM
from experimental_depth_prediction.models.DPT import DPTFineTune
from experimental_depth_prediction.models.TemporalTransformer import TemporalTransformer, DoubleTemporalTransformer, VAETemporalTransformer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import scipy


def get_model(model_type, config, device = None):
    """
    Build an untrained model of the requested type from a run config.

    Args:
        model_type (str): Model name as used in the run configs, e.g. 'ViT', 'UNet', 'DPT' or
            'temp_transformer'.
        config (dict): Run config. Reads pretraining, img_shape, sdf and the architecture
            hyperparameters of the chosen model type.
        device (str or torch.device, optional): Passed to the temporal transformer models.

    Returns:
        torch.nn.Module: The model.
    """
    pretrained = config['pretraining']
    img_shape = config['img_shape'] if 'img_shape' in config else 64
    end_features = img_shape*img_shape
    sdf = config['sdf'] if 'sdf' in config else False

    if model_type == 'ViT':
        if sdf:
            num_features = end_features
        else:
            num_features = 2*(end_features)
        if pretrained:
            model = VitFineTune(num_features = num_features, 
                                embed_only = True, 
                                pretrained = pretrained)#, embed_only = True)
        else:
            if 'smaller_vit' in config:
                if config['smaller_vit']:
                    print(f'Using smaller ViT model, dim = {config["dim"]}, \
                          num_heads = {config["num_heads"]}, \
                          ff_dim = {config["ff_dim"]}, \
                          transformer_layers = {2*config["transformer_layers"]}')
                    model = VitFromScratch(num_features = num_features, 
                                        embed_only = True, 
                                        pretrained = pretrained, 
                                        n_layers = 2*config['transformer_layers'],
                                        ff_dim = config['ff_dim'],
                                        num_heads = config['num_heads'],
                                        dim = config['dim'])
                else:
                    model = VitFromScratch(num_features = num_features, 
                                        embed_only = True, 
                                        pretrained = pretrained)
            else:
                model = VitFromScratch(num_features = num_features, 
                                    embed_only = True, 
                                    pretrained = pretrained)
        
    elif model_type == 'CNN':
        model = CNN(in_channels=1, num_features=2*end_features)
        sequence_type = 'mean'
    elif model_type == 'CNNLSTM':
        model = CNNLSTM(num_features=2*end_features)
    elif model_type == 'ResNet':
        model = ResNet(dim = 32)
    elif model_type == 'UNet':
        n_classes = 1 if sdf else 2
        if 'unet_size' in config:
            print(f"USING UNET OF SIZE {config['unet_size']}")
            model = UNet(n_channels=1, n_classes=n_classes, model_size = config['unet_size'])
        else:
            print("USING DEFAULT UNET")
            model = UNet(n_channels=1, n_classes=n_classes)
        # model = UNet(n_channels=1, n_classes=n_classes, model_size = config['unet_size'])
        repeat = False
    elif model_type == 'UNet_concat':
        n_classes = 1 if sdf else 2
        if 'unet_size' in config:
            model = UNet_concat(n_channels = config['seq_length'], n_classes = n_classes, model_size = config['unet_size'])

        else:
            model = UNet_concat(n_channels = config['seq_length'], n_classes = n_classes)
    elif model_type == 'DPT':
        model = DPTFineTune()
    elif model_type == 'temp_transformer':
        transformer_layers = config['transformer_layers']
        dim = config['dim']
        num_heads = config['num_heads']
        ff_dim = config['ff_dim']
        dropout = config['dropout']
        backbone = config['backbone']
        seq_length = config['seq_length']
        # if 'sdf' in config:
    
        sdf  = config['sdf'] if 'sdf' in config else False

        if sdf:
            end_dimensions = end_features
        else:
            end_dimensions = 2*end_features
        encoding = config['encoding'] if 'encoding' in config else 'learnable'
        if backbone == 'VAE':
            if 'vae_path' not in config or config['vae_path'] is None:
                # train VAE from scratch
                raise ValueError('VAE path not provided for Temporal Transformer')
            else:
                model = VAETemporalTransformer(transformer_layers = transformer_layers,
                                        vae_path= config['vae_path'],
                                        dim = dim,
                                        num_heads = num_heads,
                                        seq_len = seq_length,
                                        ff_dim = ff_dim,
                                        dropout = dropout, backbone = backbone,
                                        end_dimensions = end_dimensions,
                                        image_size = config['padding'], 
                                        encoding = encoding)
        else:
            model = TemporalTransformer(transformer_layers = transformer_layers,
                                        dim = dim,
                                        num_heads = num_heads,
                                        seq_len = seq_length,
                                        ff_dim = ff_dim,
                                        dropout = dropout, backbone = backbone,
                                        end_dimensions = end_dimensions,
                                        image_size = config['padding'],
                                        encoding = encoding, device = device)
        
    elif model_type == 'double_temp_transformer':
        transformer_layers = config['transformer_layers']
        dim = config['dim']
        num_heads = config['num_heads']
        ff_dim = config['ff_dim']
        dropout = config['dropout']
        backbone = config['backbone']
        seq_length = config['seq_length']

        if sdf:
            end_dimensions = end_features
        else:
            end_dimensions = 2*end_features
        model = DoubleTemporalTransformer(transformer_layers = transformer_layers, 
                                dim = dim, 
                                num_heads = num_heads, 
                                seq_len = seq_length, 
                                ff_dim = ff_dim, 
                                dropout = dropout, backbone = backbone,
                                end_dimensions = end_dimensions)
    return model

def get_dataset(config, split, model_type = None, root_folder = '', p_v_subset = None):
    """
    Build the dataset of one split from a run config.

    Picks the dataset class from the config: Flow3DDataset if pretrain_flow3d, ETDataset if
    pretrain_ET, otherwise ContourAreaDataset. Sets config['repeat'] as a side effect. The data
    folders are always built under root_folder, so thermal_root_folder and labels_folder in the
    config are not read.

    Args:
        config (dict): Run config. The dataset keyword arguments are read from it.
        split (str): Name of the split folder to load, e.g. 'train', 'test' or 'sample'.
        model_type (str, optional): Decides whether frames are repeated to 3 channels. Defaults to
            config['model_type'].
        root_folder (str): Folder that contains data/. The data folders are built under it.
        p_v_subset (float, optional): Fraction of the process conditions to keep, chosen at random.
            None keeps all.

    Returns:
        BaseContourAreaDataset: The dataset of the split.
    """
    if model_type is None:
        model_type = config['model_type']
    repeat = True
    if model_type in ('UNet', 'UNet_concat', 'DPT'):
        repeat = False
    elif model_type == 'temp_transformer':
        if config['sequence_type'] == 'segment':
            repeat = False
    config['repeat'] = repeat
    pretrain_flow3d = config.get('pretrain_flow3d', False)
    pretrain_ET = config.get('pretrain_ET', False)
    sequence_folder = config.get('sequence_folder', 'sequence_exposure')

    # Kept out of config, so saved and logged configs don't contain this machine's paths
    if pretrain_flow3d:
        thermal_root_folder = os.path.join(root_folder, 'data/Flow3D/temp_full_smoothed')
        labels_folder = os.path.join(root_folder, 'data/Flow3D/sdf_files/sdf_rolled_remade')
    elif pretrain_ET:
        thermal_root_folder = os.path.join(root_folder, 'data/ET_sequence/temp_full_smoothed')
        labels_folder = os.path.join(root_folder, 'data/ET_sequence/depths_full_smoothed')
    else:
        thermal_root_folder = os.path.join(root_folder, f"data/{sequence_folder}/")
        labels_folder = os.path.join(root_folder, 'data/input_label_map')

    dataset_configuration = {'thermal_root_folder': thermal_root_folder,
                             'labels_folder': labels_folder}
    dataset_configuration_keys = [
        'padding',
        'moving_average_window_size',
        'exposure',
        'repeat',
        'sequence_type',
        'image_type',
        'sequence_length',
        'moving_average_window_stride',
        'augment_transforms',
        'randomize_contours',
        'model_type',
        'subset',
        'label_file',
        'sdf',
        'compress_temperatures',
        'noise_factor'
    ]
    for key in dataset_configuration_keys:
        if key == 'sequence_length':
            dataset_configuration[key] = config['seq_length']
        elif key in config:
            dataset_configuration[key] = config[key]

    if pretrain_flow3d:
        dataset_object = Flow3DDataset
    elif pretrain_ET:
        dataset_object = ETDataset
    else:
        dataset_object = ContourAreaDataset
    return dataset_object(split = split, p_v_subset = p_v_subset, **dataset_configuration)


def get_dataloader(batch_size, config, model_type, root_folder = '', label_file = 'input_label_map', return_dev = False):
    """
    Build the train and test dataloaders, and optionally the validation one, from a run config.

    Each split is built with get_dataset, so the same side effects on config apply.

    Args:
        batch_size (int): Batch size of every dataloader.
        config (dict): Run config. The dataset keyword arguments are read from it.
        model_type (str): Decides whether frames are repeated to 3 channels.
        root_folder (str): Folder that contains data/. The data folders are built under it.
        label_file (str): Not read. The label file comes from config['label_file'].
        return_dev (bool): Whether to also return the validation dataloader. The validation split is
            'validation', except for the 'sequence_exposure' folder, which has none, so 'test' is used.

    Returns:
        tuple: (train_dataloader, test_dataloader), plus dev_dataloader if return_dev.
    """
    # Used to select smaller portion of data for training
    dataset_ratio = config.get('dataset_ratio')
    train_dataset = get_dataset(config, 'train', model_type, root_folder, p_v_subset = dataset_ratio)
    print('Sample shape:', train_dataset[0][0].shape)
    test_dataset = get_dataset(config, 'test', model_type, root_folder)

    train_dataloader =  DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True, num_workers = 1)

    test_dataloader =  DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True, num_workers = 1)

    print(len(train_dataloader), "training batches available")
    if return_dev:
        sequence_folder = config.get('sequence_folder', 'sequence_exposure')
        dev_split = 'test' if sequence_folder == 'sequence_exposure' else 'validation'
        dev_dataset = get_dataset(config, dev_split, model_type, root_folder)
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=batch_size,
                                    shuffle=False, num_workers = 1)
        return train_dataloader, test_dataloader, dev_dataloader
    return train_dataloader, test_dataloader


def load_model(model_type,model_artifact_path, config, run_path = None,  device = 'cuda', train = False):
    """
    Load a trained model from a wandb artifact or a local checkpoint.

    Args:
        model_type (str): Passed to get_model.
        model_artifact_path (str, optional): A wandb artifact name containing ':' (downloaded through
            the API; a ':v2' that does not exist falls back to ':v0'), or a local checkpoint path.
            Built from run_path if None.
        config (dict): Run config, passed to get_model.
        run_path (str, optional): wandb run path used to build the artifact name when
            model_artifact_path is None.
        device (str): Device the checkpoint and model are loaded on. The CPU is used when CUDA is
            unavailable.
        train (bool): Whether to leave the model in train mode instead of eval mode.

    Returns:
        torch.nn.Module: The model with the checkpoint weights loaded. A 'module.' prefix left by
            DataParallel is stripped from the state dict keys if needed.

    Raises:
        Exception: If neither model_artifact_path nor run_path is given.
    """

    if model_artifact_path is None:
        if run_path is None:
            raise Exception('No model checkpoint or WandB run path specified.')
        entity, project, run_id = run_path.split('/')
        model_artifact_path = f'{entity}/{project}/{model_type}-{run_id}:v0'
    
    saved_model_artifact = model_artifact_path

    if ':' in model_artifact_path:
        api = wandb.Api()
        try:
            artifact = api.artifact(saved_model_artifact, type = 'model')
        except wandb.errors.CommError:
            artifact = api.artifact(saved_model_artifact.replace(':v2', ':v0'), type = 'model')
        # artifact = wandb.run.use_artifact(saved_model_artifact, type='model')
        artifact_dir = artifact.download()
        if not torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(artifact_dir, 'model.pt'), map_location = 'cpu')
        else:
            checkpoint = torch.load(os.path.join(artifact_dir, 'model.pt'), map_location = device)
    else:
        checkpoint = torch.load(model_artifact_path, map_location = device)
    model = get_model(model_type, config, device = device)
    try:
        model.load_state_dict(checkpoint[0])
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in checkpoint[0].items():
            name = k[len('module.'):] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.to(device)
    
    # model = nn.DataParallel(model, device_ids = range(0, 4))
    if train:
        model.train()
    else:
        model.eval()
    # print(torchsummary.summary(model.cuda(),  (1,64,64)))
    return model

def rescale_image(experimental_image, flow3d_px = 10, experimental_px = 5.6, image_size = 64):
    """
    Resample an experimental frame to the FLOW-3D pixel size and pad or crop it to image_size.

    Args:
        experimental_image (numpy.ndarray): Frame with the two spatial axes last.
        flow3d_px (float): FLOW-3D pixel size in µm.
        experimental_px (float): Camera pixel size in µm.
        image_size (int): Side length of the output in pixels.

    Returns:
        numpy.ndarray: The resampled frame, zero padded if it became smaller than image_size and
            center cropped if larger.
    """

    zooms = np.ones(len(experimental_image.shape))
    zooms[-2:] = experimental_px/flow3d_px
    zoomed = scipy.ndimage.zoom(experimental_image, zoom = zooms, order = 1)
    if experimental_px < flow3d_px:
        left_pad = (image_size - zoomed.shape[0])//2
        right_pad = image_size - zoomed.shape[0] - left_pad
        top_pad = (image_size - zoomed.shape[1])//2
        bottom_pad = image_size - zoomed.shape[1] - top_pad
        padded_zoomed_image = np.pad(zoomed, ((left_pad, right_pad), (top_pad, bottom_pad)), mode = 'constant', constant_values = 0)
    else:
        left_bound = zoomed.shape[-2]//2 - image_size//2
        right_bound = zoomed.shape[-2]//2 + (image_size - image_size//2)
        top_bound = zoomed.shape[-1]//2 - image_size//2
        bottom_bound = zoomed.shape[-1]//2 + (image_size - image_size//2)
        padded_zoomed_image  = np.take(zoomed, indices = (np.arange(left_bound, right_bound)), axis = (len(zoomed.shape) - 2))
        padded_zoomed_image = np.take(padded_zoomed_image, indices = (np.arange(top_bound, bottom_bound)), axis = len(zoomed.shape) - 1)

    return padded_zoomed_image

def convert_to_flow3d(experimental):
    """Swap the two spatial axes of a batch of experimental frames and return it as a NumPy array."""
    axes = np.arange(len(experimental.shape))
    temp = axes[-1]
    axes[-1] = axes[-2]
    axes[-2] = temp
    experimental_transposed = np.transpose(experimental.cpu().detach().numpy(), axes = axes)
    # experimental_transposed = rescale_image(experimental_transposed, flow3d_px = 5)
    return experimental_transposed

def plot_dimensionality_reduction(low_dim, mask, title, color, highlight_pv, highlight_time, ax = None, show = False, results_folder = None):
    """
    Scatter plot of 2D embedded samples, colored by energy density.

    With highlight_pv, samples near that energy density are drawn opaque with black edges and the
    rest faded. With highlight_time as well, one of the highlighted samples gets a red edge and the
    title shows its frame number.

    Args:
        low_dim (numpy.ndarray): (N, 2) embedded samples.
        mask (numpy.ndarray): (N,) boolean mask of the samples to plot.
        title (str): Figure title and, when saving, the file name.
        color (array): (N,) energy density of each sample.
        highlight_pv (str, optional): Process condition key "power_velocity" whose energy density is
            highlighted.
        highlight_time (int, optional): Index within the highlighted samples to mark in red.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure is created if None.
        show (bool): Whether to show the figure. If False it is saved as <results_folder>/<title>.png,
            logged to wandb and closed.
        results_folder (str, optional): Folder for the PNG when show is False.
    """

    if ax is None:
        plt.figure(dpi = 150, figsize = np.array([4,3])*1.3)
        ax = plt.gca()
    
    low_dim = np.array(low_dim)[mask]
    color = np.array(color)[mask]
    
    alpha = 1
    edgecolor = 'k'
    if highlight_pv is not None:
        alpha = 0.25
        specific_density = float(highlight_pv.split('_')[0])/float(highlight_pv.split('_')[1])
        edgecolor = "None"
    ax.set_title(title)
    ax.scatter(low_dim[:,0], low_dim[:,1], edgecolor = edgecolor,c = np.array(color), alpha=alpha)
    if highlight_pv is not None:
        density_mask = np.abs(np.array(color) - specific_density) < 0.1
        ax.scatter(low_dim[:,0][density_mask], low_dim[:,1][density_mask], edgecolor = 'k',c = np.array(color)[density_mask], alpha=0.75)
        if highlight_time is not None:
            ax.scatter(low_dim[:,0][density_mask][highlight_time], low_dim[:,1][density_mask][highlight_time], edgecolor = 'r',c = np.array(color)[density_mask][highlight_time], alpha=1)
    
    ax.set_xlabel('PC Dimension 1')
    ax.set_ylabel('PC Dimension 2')
    if highlight_time is not None:
        ax.set_title("Current Frame: {:03d}".format(highlight_time))
    if show:
        frame_tick()
        plt.show()
    else:
        plt.savefig(os.path.join(results_folder, f'{title}.png'))
        wandb.log({f"{title}": wandb.Image(os.path.join(results_folder, f'{title}.png'))})
        plt.clf()
        plt.close('all')


def dimensionality_reduction(pv_thermal_image, highlight_pv = None):
    """
    Fit a 2-component PCA to the thermal images of every process condition.

    Each image is flattened and centered, then divided by its per-pixel standard deviation. Pixels
    with a standard deviation below 100 are left unscaled.

    Args:
        pv_thermal_image (dict): Stack of thermal images for each process condition key, as returned
            by get_pv_thermal_image(). Empty entries are skipped.
        highlight_pv (str, optional): Not read.

    Returns:
        tuple: (low_dim, densities). low_dim is the (N, 2) PCA projection and densities the energy
            density, power divided by velocity, of each row.
    """


    data_samples = []
    densities = []
    for key in pv_thermal_image.keys():
        if len(pv_thermal_image[key])  == 0:
            continue
        copy_pv_thermal_image = np.nan_to_num(np.array(pv_thermal_image[key]))

        data_samples.append(copy_pv_thermal_image)
      
        mean_images = np.nanmean(np.array(pv_thermal_image[key]), axis=(1,2))
        power = np.array(key.split('_')).astype(float)[0]
        velocity = np.array(key.split('_')).astype(float)[1]
        densities.extend([power/velocity]*len(mean_images))

    # perform pca on data_samples
    data_array = np.concatenate(data_samples)
   
    pca = PCA(n_components=2)   
    data_feature_matrix = data_array.reshape(data_array.shape[0], -1)
    # normalize data_feature_matrix
    data_feature_matrix = (data_feature_matrix - np.mean(data_feature_matrix, axis=0))
    std_matrix = np.std(data_feature_matrix, axis=0)

    std_matrix[np.abs(std_matrix) < 100] = 1
    data_feature_matrix = data_feature_matrix / std_matrix
    pca.fit(data_feature_matrix)
    low_dim_data = pca.transform(data_feature_matrix)
    return low_dim_data, densities

def plot_input_pca(dataset, title, results_folder):
    """
    Plot a 2D PCA of a dataset's thermal image sequences, one point per sample, colored by energy
    density.

    Args:
        dataset (BaseContourAreaDataset): Dataset to project. process_parameter_mapping is called on
            it.
        title (str): Figure title and file name.
        results_folder (str): Folder the PNG is saved in.
    """
    dataset.process_parameter_mapping()

    low_dim, densities = dimensionality_reduction(dataset.get_pv_thermal_image())

    plot_dimensionality_reduction(low_dim,
                                  mask = np.ones(len(low_dim), dtype = bool),
                                  title = title,
                                  color = densities,
                                  highlight_pv = None,
                                  highlight_time = None,
                                  results_folder = results_folder)

def create_config(run_path, sweep_id = ''):
    """
    Download a wandb run's config and save it as results[_<sweep_id>]/<run id>/config/config.json.

    Args:
        run_path (str): wandb run path, "<entity>/<project>/<run id>".
        sweep_id (str): If given, the results folder is results_<sweep_id> instead of results.

    Returns:
        tuple: (run, config, config_directory). The wandb Run object, its config dict and the folder
            the JSON was written to.
    """

    api = wandb.Api()
    run = api.run(run_path)
    config = run.config
    config_directory = os.path.join(f'results_{sweep_id}', run_path.split('/')[-1], 'config')
    os.makedirs(config_directory, exist_ok = True)
    json.dump(config, open(os.path.join(f"{config_directory}", "config.json"), 'w' ) )
    return run, config, config_directory

def dict2namespace(config):
    """Convert a nested dict into nested argparse.Namespace objects."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def compute_iou(gt, pred, plot = False, ax = None):
    """
    Compute the intersection over union of two binary masks.

    Args:
        gt (numpy.ndarray): Ground-truth mask.
        pred (numpy.ndarray): Predicted mask.
        plot (bool): Whether to draw the intersection and union on ax.
        ax (matplotlib.axes.Axes, optional): Axes for the plot. Defaults to the current axes.

    Returns:
        float: IoU. NaN when both masks are empty.
    """
    # plt.figure(dpi = 150)

    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.imshow(intersection.astype(float) + union.astype(float))
        # clb = plt.colorbar()
        # clb.set_ticks([0,1,2])
        # clb.set_ticklabels(['None', 'Intersection', 'Union'])
        ax.set_title(f"IoU = {np.sum(intersection) / np.sum(union):.2f}")
        plt.axis('off')
        frame_tick()
        if ax is None:
            plt.show()
    return np.sum(intersection) / np.sum(union)

def resample_sdf(sdf, new_resolution = 1024):
    """
    Resize a signed distance field to a new square resolution with cubic interpolation.

    Args:
        sdf (numpy.ndarray): (H, W) or (N, H, W) signed distance field.
        new_resolution (int): Output side length.

    Returns:
        numpy.ndarray: The resized field, (new_resolution, new_resolution) or
            (N, new_resolution, new_resolution).
    """
    if len(sdf.shape) > 2:
        new_sdf = skimage.transform.resize(sdf, (sdf.shape[0], new_resolution, new_resolution), order = 3)
    else:
        new_sdf = skimage.transform.resize(sdf, (new_resolution, new_resolution), order = 3)
    return new_sdf

def sdf_to_binary(sdf, resolution  = 64):
    """
    Resample a signed distance field and threshold it into a binary mask.

    Args:
        sdf (numpy.ndarray): (H, W) or (N, H, W) signed distance field.
        resolution (int): Side length to resample to before thresholding.

    Returns:
        numpy.ndarray: Mask with 1 where the distance is negative.
    """
    new_sdf = resample_sdf(sdf, new_resolution = resolution)
    return new_sdf < 0