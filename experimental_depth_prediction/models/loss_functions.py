import torch.nn as nn


def calculate_loss(output, target, method='SDF'):
    """
    Compute the training loss for one batch.

    Args:
        output (torch.Tensor): Model output, (N, H*W) for SDF or (N, 2*H*W) for binary.
        target (torch.Tensor): Labels, (N, 1, H, W) or (N, H, W). Signed distances for SDF, 0/1
            for binary.
        method (str): 'SDF' uses mean squared error on the distances. Anything else uses two-class
            cross-entropy over the flattened pixels.

    Returns:
        torch.Tensor: Scalar loss.
    """
    
    if method == 'SDF':
        criterion = nn.MSELoss()
   
    else:
        criterion = nn.CrossEntropyLoss()

    reshaped_output = output.reshape(target.shape[0], -1)
    reshaped_target = target.reshape(target.shape[0], -1)
    # SDF Loss Calculation
    if method == 'SDF':
        loss = criterion(reshaped_output.float(), reshaped_target.float())
    else:
        reshaped_output = output.reshape(target.shape[0], 2, -1).float()
        
        loss = criterion(reshaped_output.float(), reshaped_target.long())
  
    return loss