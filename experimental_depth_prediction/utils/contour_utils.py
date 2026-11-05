import numpy as np
from scipy.interpolate import interp1d
from skimage import measure
from experimental_depth_prediction.utils.utils import sdf_to_binary

# Physical size of the cross-section field of view, in µm. Every pixel -> µm conversion in the
# repo (training plots, analysis metrics) goes through pixel_size() so the numbers agree.
IMAGE_EXTENT_X = 368.208
IMAGE_EXTENT_Y = 450.65


def pixel_size(side_length):
    """
    Return the pixel size of a square cross-section in µm.

    Assumes the cross-section spans the full field of view (IMAGE_EXTENT_Y by IMAGE_EXTENT_X)
    at any resolution.

    Args:
        side_length (int): Side length of the cross-section in pixels.

    Returns:
        tuple: (dy, dx) in µm per pixel.
    """
    return IMAGE_EXTENT_Y / side_length, IMAGE_EXTENT_X / side_length


def predictions_to_cross_sections(image, 
                      method,  
                      img_size = None,
                      conversion = 'coarse'):
    """
    Convert raw model output into binary melt pool cross-sections.

    Args:
        image (torch.Tensor): Model output for a batch, (N, C*H*W) or (N, C, H, W). C is 2 for
            cross-entropy and 1 for SDF.
        method (str): 'SDF' or 'binary'.
        img_size (int or tuple of int, optional): Output size as (height, width), or one int for a
            square image. Inferred as square from the output length if not given.
        conversion (str): SDF only. 'coarse' thresholds at the output resolution. Any other value
            upsamples to 256x256 first.

    Returns:
        numpy.ndarray: (N, H, W) cross-sections with 1 for melt pool and 0 for background, or
            (N, 256, 256) for SDF with a conversion other than 'coarse'.

    Raises:
        ValueError: If img_size is neither an int nor a length-2 tuple or list.
    """
    # SDF: 1 channel, Dice: 1 channel, Cross-entropy: 2 channels
    n_channels = {'SDF': 1, 'binary': 2}[method]
    
    image = image.detach().cpu().reshape(image.shape[0], n_channels, -1)

    if img_size is not None:
        # If img_size is provided, use it to reshape the image
        if isinstance(img_size, int):
            img_shape = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_shape = tuple(img_size)
        else:
            raise ValueError("img_size must be an int or a tuple/list of length 2.")
    else:
        side_length = int(round(np.sqrt(image.shape[-1]))) # assumes image is square
        img_shape = (side_length, side_length)

    height, width = img_shape
    reshaped_image = image.reshape(image.shape[0], n_channels, height, width)

    if method == 'SDF':
        sdf = reshaped_image[:, 0].numpy()
        if conversion == 'coarse':
            binary = (sdf < 0).astype('float')
        else:
            binary = sdf_to_binary(sdf, resolution = 256)
    else:
        binary = reshaped_image.argmax(dim = 1).numpy()
    return binary

def targets_to_cross_sections(targets, method):
    """
    Reshape labels to (N, H, W) binary cross-sections.

    Binary labels are already 0/1 and are only reshaped. SDF labels are signed distances, so pixels
    with a negative distance become 1 (melt pool) and the rest 0.

    Args:
        targets (torch.Tensor): Labels for a batch, (N, 1, H, W) or (N, H, W).
        method (str): 'SDF' or 'binary'.

    Returns:
        numpy.ndarray: (N, H, W) cross-sections.
    """
    targets = targets.detach().cpu().reshape(targets.shape[0], targets.shape[-2], targets.shape[-1]).numpy()
    if method == 'SDF':
        return (targets < 0).astype('float')
    return targets


def resample_contour(contour, n_points = 100):
    """
    Close a contour and resample it to equally spaced points along its length.

    Args:
        contour (numpy.ndarray): (N, 2) points.
        n_points (int): Number of output points.

    Returns:
        numpy.ndarray: (n_points, 2) points.
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    distance = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(contour, axis = 0), axis = 1))])
    new_distance = np.linspace(0, distance[-1], n_points)
    return interp1d(distance, contour, axis = 0)(new_distance)


def contour_um(cross_section):
    """
    Extract the outline of a binary cross-section in µm.

    When there are several outlines the longest is used.

    Args:
        cross_section (numpy.ndarray): (H, W) binary cross-section.

    Returns:
        numpy.ndarray or None: (100, 2) outline points as (depth, width) in µm, or None if the
            cross-section is empty.
    """
    outlines = measure.find_contours(cross_section.astype(float), 0.5)
    if not outlines:
        return None
    longest = max(outlines, key = len)
    if len(longest) < 2:
        return None
    return resample_contour(longest * np.array(pixel_size(cross_section.shape[-1])))


def depth_px(cross_section):
    """Return the index of the deepest melted row of a cross-section, or 0 if nothing is melted."""
    rows = np.where(cross_section)[0]
    return rows.max() if rows.size else 0
