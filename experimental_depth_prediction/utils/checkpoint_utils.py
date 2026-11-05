import os
import torch
import wandb


def checkpoint_path(results_folder, metric):
    """Return the path of the best-<metric> checkpoint in results_folder."""
    return os.path.join(results_folder, f'ckpt_best_{metric}.pth')


def upload_checkpoint(path, model_type, metric):
    """Log a checkpoint file to wandb as a model artifact aliased best_<metric>."""
    art = wandb.Artifact(f'{model_type}-{wandb.run.id}', type = 'model')
    art.add_file(path, 'model.pt')
    wandb.log_artifact(art, aliases = [f'best_{metric}'])


def save_if_best(metric, value, best, model, optimizer, epoch, results_folder, model_type, log_model_frequency):
    """
    Record and checkpoint a new best value of a test metric.

    Lower is better for 'loss'; higher is better for anything else. NaN never counts as better.

    Args:
        metric (str): 'loss', 'iou' or 'r2'.
        value (float): this epoch's value of the metric.
        best (dict): best values so far, keyed by metric; updated in place.
        model, optimizer: saved as [model.state_dict(), optimizer.state_dict(), epoch].
        epoch (int): current epoch.
        results_folder (str): where ckpt_best_<metric>.pth is written (overwritten on improvement).
        model_type (str): used in the wandb artifact name.
        log_model_frequency (int): 0 = no checkpoints, 1 = save to disk, 2+ = also upload on every
            improvement.
    """
    improved = value < best[metric] if metric == 'loss' else value > best[metric]
    if not improved:
        return
    best[metric] = value
    wandb.run.summary[f'best_{metric}'] = value
    wandb.run.summary[f'best_{metric}_epoch'] = epoch
    if log_model_frequency > 0:
        path = checkpoint_path(results_folder, metric)
        torch.save([model.state_dict(), optimizer.state_dict(), epoch], path)
        if log_model_frequency > 1:
            upload_checkpoint(path, model_type, metric)


def upload_best_checkpoints(results_folder, model_type):
    """Upload the final best checkpoint for each metric, skipping any that were never saved."""
    for metric in ['iou', 'loss', 'r2']:
        path = checkpoint_path(results_folder, metric)
        if os.path.exists(path):
            upload_checkpoint(path, model_type, metric)
        else:
            print(f'No checkpoint saved for best {metric}, skipping upload')
