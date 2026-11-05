import numpy as np

def load_image(fname, padding, repeat):
    """
    Load a .npy frame and center crop it to a square of side 2 * padding.

    The frame is edge padded by padding pixels first, so the crop never runs past the image.

    Args:
        fname (str): Path of the .npy file.
        padding (int): Half the side length of the crop.
        repeat (bool): Whether to copy the frame to 3 channels.

    Returns:
        numpy.ndarray: The crop, (H, W) or (3, H, W) with repeat.
    """
    image_npy = np.load(fname)
    p = padding
    padded_image = np.pad(image_npy, mode='edge', pad_width=p)
    x_midpoint = int(np.round(padded_image.shape[0] / 2))
    y_midpoint = int(np.round(padded_image.shape[1] / 2))
    cropped_image = padded_image[x_midpoint - p:x_midpoint + p, y_midpoint - p:y_midpoint + p]
    image_npy = cropped_image
    if repeat:
        image_npy = np.repeat(image_npy[None], 3, 0)
    return image_npy
