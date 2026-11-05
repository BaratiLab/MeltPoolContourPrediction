
import os
import numpy as np
import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_squared_error, r2_score
from experimental_depth_prediction.utils.utils import  compute_iou
from experimental_depth_prediction.utils.utils import convert_to_flow3d
from experimental_depth_prediction.models.loss_functions import calculate_loss
from experimental_depth_prediction.utils.contour_utils import predictions_to_cross_sections, targets_to_cross_sections, pixel_size
from experimental_depth_prediction.utils.data_utils import parse_process_param, energy_density
from experimental_depth_prediction.utils.plotting_utils import plot_epoch
# matplotlib.use('Agg') 
from experimental_depth_prediction.utils.checkpoint_utils import save_if_best, upload_best_checkpoints


def run_epoch(model, 
              dataloader, 
              device, 
              method, 
              optimizer, 
              resample_to_flow3d = False):
    """
    Run one pass over a dataloader. Train if an optimizer is given, otherwise evaluate.

    Training puts the model in train mode, records gradients and steps the optimizer after each
    batch. Evaluation puts the model in eval mode and records no gradients.

    Args:
        model (torch.nn.Module): The model, already on device.
        dataloader (torch.utils.data.DataLoader): Yields (image, target, index) batches.
        device (torch.device): Device the batches are moved to.
        method (str): 'SDF' or 'binary'. Selects the loss and how outputs become cross-sections.
        optimizer (torch.optim.Optimizer or None): Optimizer to train with, or None to evaluate.
        resample_to_flow3d (bool): Whether to resample images to the FLOW-3D layout first.

    Returns:
        dict:
            'loss' (float): Mean loss per batch.
            'iou' (float): Mean of the per-batch mean IoUs, ignoring NaN batches.
            'predictions', 'labels' (list of (H, W) arrays): Predicted and label cross-sections, one
                per sample, in dataloader order.
            'powers', 'velocities' (numpy.ndarray, (N,)): Process parameters per sample in the same
                order. Velocities are in m/s for experimental keys and mm/s for FLOW-3D keys.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    ious, prediction_cross_sections, label_cross_sections, powers, velocities =  [], [], [], [], []

    running_loss = 0

    for image, target, index in tqdm(dataloader, disable = not training, desc = 'Training' if training else 'Evaluating'):
        image = image.to(device)
        target = target.to(device)

        if resample_to_flow3d:
            image = torch.tensor(convert_to_flow3d(image)).to(device)

        for sample_id in index:
            pv_string = dataloader.dataset.__getiteminfo__(sample_id.item())[0]['pv']
            power, velocity = parse_process_param(pv_string)
            powers.append(power)
            velocities.append(velocity)

        with torch.set_grad_enabled(training):
            output = model((image.float()))
            loss = calculate_loss(output, 
                                target, 
                                method = method)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()/len(dataloader)

        pred_cross_section_batch = predictions_to_cross_sections(output, 
                                                            method)
        target_cross_section_batch = targets_to_cross_sections(target, method)
        iou_batch = np.nanmean([compute_iou(target_cross_section_batch[i], pred_cross_section_batch[i]) for i in range(len(target_cross_section_batch))])

        prediction_cross_sections.extend(pred_cross_section_batch)
        label_cross_sections.extend(target_cross_section_batch)

        ious.append(iou_batch)

    return dict(loss = running_loss, 
                iou = np.nanmean(ious), 
                predictions = prediction_cross_sections, 
                labels = label_cross_sections,
                powers = np.array(powers),
                velocities = np.array(velocities))

def train(model, 
        train_dataloader, 
        test_dataloader, 
        dev_dataloader, 
        config_dictionary, 
        results_folder, 
        method = 'SDF',
        weight_decay=0,
        use_scheduler=False,
        resample_to_flow3d = False,
        device = None):
    """
    Train a model and log metrics and checkpoints to wandb.

    Each epoch runs run_epoch on the training set with the optimizer and on dev_dataloader without
    it, logs loss, IoU, area MSE and area R² for both, and checkpoints the model whenever the
    validation loss, IoU or R² improves. Every 5 epochs the monitoring plots are logged with
    plot_epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_dataloader (torch.utils.data.DataLoader): Training set.
        test_dataloader (torch.utils.data.DataLoader): Test set. Not used during training.
        dev_dataloader (torch.utils.data.DataLoader): Validation set, used for the logged metrics and
            for checkpoint selection.
        config_dictionary (dict): Needs 'epochs', 'learning_rate', 'model_type' and
            'model_log_frequency'. model_log_frequency is 0 for no checkpoints, 1 to save them to
            disk and 2 to also upload them to wandb on every improvement.
        results_folder (str): Folder for the loss text files, checkpoints and plot PNGs.
        method (str): 'SDF' or 'binary'. Passed to run_epoch.
        weight_decay (float): Weight decay of the Adam optimizer.
        use_scheduler (bool): Whether to halve the learning rate when the validation loss plateaus.
        resample_to_flow3d (bool): Passed to run_epoch.
        device (torch.device, optional): Device to train on. For a DataParallel model, its first GPU.
            Defaults to CUDA if available, otherwise the CPU.
    """
    epochs = config_dictionary['epochs']
    log_model_frequency = config_dictionary['model_log_frequency']
    
    learning_rate = config_dictionary['learning_rate']
    model_type = config_dictionary['model_type']
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.5)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    losses = []
    test_losses = []



    best = {'loss': float('inf'), 'iou': -float('inf'), 'r2': -float('inf')}

    wandb.watch(model, log_freq = 10)
    for epoch in tqdm(range(epochs)):

        train_result = run_epoch(model, 
            train_dataloader, 
            device, 
            method, 
            optimizer = optimizer,
            resample_to_flow3d = resample_to_flow3d, 
           )
        
        losses.append(train_result['loss'])
        test_result = run_epoch(model, 
            dev_dataloader, 
            device, 
            method, 
            optimizer = None,
            resample_to_flow3d = resample_to_flow3d, 
            )
        test_losses.append(test_result['loss'])
        pixel_area = np.prod(pixel_size(train_result['predictions'][0].shape[-1]))


        for split in (train_result, test_result):
            split['true_areas'] = np.array([c.sum() for c in split['labels']]) * pixel_area
            split['pred_areas'] = np.array([c.sum() for c in split['predictions']]) * pixel_area
            split['energy_density'] = energy_density(split['powers'], split['velocities'])
        train_mse = mean_squared_error(train_result['true_areas'], train_result['pred_areas'])
        train_r2 = r2_score(train_result['true_areas'], train_result['pred_areas'])
        test_mse = mean_squared_error(test_result['true_areas'], test_result['pred_areas'])
        test_r2 = r2_score(test_result['true_areas'], test_result['pred_areas'])

        wandb.log({"train_loss": train_result['loss'], "test_loss": test_result['loss'],
                    "train_iou": train_result['iou'], "test_iou": test_result['iou'],
                    "train_area_mse": train_mse, "test_area_mse": test_mse,
                    "train_area_r2": train_r2, "test_area_r2": test_r2}, step = epoch)
        if use_scheduler:
            scheduler.step(test_result['loss'])
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step = epoch)

        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{epochs}] Train Loss: {train_result["loss"]:.05f}, Validation Loss: {test_result["loss"]:.05f}, '
                  f'Train R^2: {train_r2:.05f}, Test R^2: {test_r2:.05f}, '
                  f'Train IOU: {train_result["iou"]:.05f}, Test IOU: {test_result["iou"]:.05f}, '
                  f'Train Area MSE: {train_mse:.05f}, Test Area MSE: {test_mse:.05f}')
            np.savetxt(os.path.join(results_folder, 'train_loss.txt'), losses)
            np.savetxt(os.path.join(results_folder, 'test_loss.txt'), test_losses)
            plot_epoch({'train': train_result, 'test': test_result}, epoch, pixel_area, results_folder)

        for metric, value in [('loss', test_result['loss']), ('iou', test_result['iou']), ('r2', test_r2)]:
            save_if_best(metric, value, best, model, optimizer, epoch, results_folder, model_type, log_model_frequency)
    if log_model_frequency > 0:
        upload_best_checkpoints(results_folder, model_type)
    wandb.finish()
