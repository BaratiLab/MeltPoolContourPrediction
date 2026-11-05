"""
Evaluate a trained cross-section model on a dataset.

evaluate() returns a pandas DataFrame with one row per sample holding the process parameters,
energy density and every metric in METRICS. summarize() reduces that table to the values written
to error_metrics.json. Plotting is done in analysis_tools_v2.py.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import r2_score

from experimental_depth_prediction.utils.utils import compute_iou
from experimental_depth_prediction.utils.contour_utils import (predictions_to_cross_sections, targets_to_cross_sections,
                                                               pixel_size, contour_um, depth_px)
from experimental_depth_prediction.utils.data_utils import parse_process_param, energy_density

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Per-sample metric columns produced by evaluate(). summarize() reports mean and std of each.
METRICS = ['iou', 'area_error', 'area_percent_error', 'depth_error', 'depth_percent_error',
           'contour_mae', 'contour_rmse', 'hausdorff']


def cross_section_metrics(target, pred):
    """
    Compute every per-sample metric for one pair of binary (H, W) cross-sections.

    Areas are in µm² and depths in µm. The contour metrics compare the two outlines after resampling
    each to 100 points and are NaN when either cross-section is empty.

    Args:
        target (numpy.ndarray): Ground-truth cross-section.
        pred (numpy.ndarray): Predicted cross-section.

    Returns:
        dict: iou, area_gt, area_pred, depth_gt, depth_pred, contour_mae, contour_rmse and hausdorff.
    """
    dy, dx = pixel_size(target.shape[-1])
    metrics = dict(iou = compute_iou(target, pred),
                   area_gt = target.sum() * dx * dy, area_pred = pred.sum() * dx * dy,
                   depth_gt = depth_px(target) * dy, depth_pred = depth_px(pred) * dy,
                   contour_mae = np.nan, contour_rmse = np.nan, hausdorff = np.nan)
    gt_contour, pred_contour = contour_um(target), contour_um(pred)
    if gt_contour is not None and pred_contour is not None:
        difference = pred_contour - gt_contour
        metrics.update(contour_mae = np.abs(difference).mean(),
                       contour_rmse = np.sqrt((difference ** 2).mean()),
                       hausdorff = max(directed_hausdorff(pred_contour, gt_contour)[0],
                                       directed_hausdorff(gt_contour, pred_contour)[0]))
    return metrics


def embed(model, image, model_type):
    """
    Extract the latent representation of a batch.

    Args:
        model (torch.nn.Module): The model, optionally wrapped in DataParallel.
        image (torch.Tensor): Input batch on the model's device.
        model_type (str): Selects which layer is read.

    Returns:
        numpy.ndarray or None: (N, D) latents, or None for model types without an exposed latent.
    """
    net = model.module if isinstance(model, nn.DataParallel) else model
    if model_type == 'DPT':
        z = net.dpt.dpt(image).last_hidden_state[:, -1]
    elif model_type == 'ViT':
        z = net.vit(net.resize(image))
    elif model_type in ('UNet', 'UNet_concat'):
        z = net.embed(image)
    elif model_type in ('temp_transformer', 'double_temp_transformer'):
        z = net.embed(image)[:, -1]
    else:
        return None
    return z.flatten(1).detach().cpu().numpy()


@torch.no_grad()
def evaluate(model, dataloader, method, img_shape = None, model_type = None, embeddings = False,
             n_keep = 0, device = DEVICE):
    """
    Run a model over a dataloader and tabulate one row per sample.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): Yields (image, target, index) batches.
        method (str): 'SDF' or 'binary'. How the output becomes a cross-section.
        img_shape (int, optional): Output side length, if it cannot be inferred as square.
        model_type (str, optional): Required when embeddings is True. Selects the latent to extract.
        embeddings (bool): Whether to add an 'embedding' column with one (D,) array per row.
        n_keep (int): Number of leading samples whose raw image, target and prediction are kept in
            df.attrs['samples'] for publication figures.
        device (str): Device the batches are moved to.

    Returns:
        pandas.DataFrame: Columns pv, frame, power, velocity, energy_density, area_gt, area_pred,
            depth_gt, depth_pred and every metric in METRICS, plus embedding if requested.
    """
    model.eval()
    dataset = dataloader.dataset
    rows, kept = [], []
    for image, target, index in tqdm(dataloader, desc = 'Evaluating'):
        image = image.to(device).float()
        output = model(image)
        preds = predictions_to_cross_sections(output, method, img_size = img_shape)
        targets = targets_to_cross_sections(target, method)
        latents = embed(model, image, model_type) if embeddings else None
        for i, sample_id in enumerate(index.tolist()):
            info = dataset.__getiteminfo__(sample_id)[0]
            power, velocity = parse_process_param(info['pv'])
            row = dict(pv = info['pv'], frame = info.get('frame_number'), power = power, velocity = velocity,
                       energy_density = energy_density(power, velocity), **cross_section_metrics(targets[i], preds[i]))
            if latents is not None:
                row['embedding'] = latents[i]
            rows.append(row)
            if len(kept) < n_keep:
                kept.append(dict(image = image[i].cpu().numpy(), target = targets[i], prediction = preds[i],
                                 power = power, velocity = velocity))
    df = pd.DataFrame(rows)
    for quantity in ('area', 'depth'):
        df[f'{quantity}_error'] = (df[f'{quantity}_pred'] - df[f'{quantity}_gt']).abs()
        df[f'{quantity}_percent_error'] = df[f'{quantity}_error'] / df[f'{quantity}_gt']
    df.attrs['samples'] = kept
    return df


def summarize(df):
    """
    Reduce a per-sample table to summary statistics.

    Returns:
        dict: Mean and population standard deviation of every metric in METRICS, plus the R² of the
            area and depth predictions.
    """
    summary = {}
    for metric in METRICS:
        summary[metric] = float(df[metric].mean())
        summary[f'{metric}_std'] = float(df[metric].std(ddof = 0))
    for quantity in ('area', 'depth'):
        summary[f'r2_{quantity}'] = float(r2_score(df[f'{quantity}_gt'], df[f'{quantity}_pred']))
    return summary
