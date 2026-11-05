import glob
import os

import numpy as np
import torch
from torchvision.transforms import Resize

from experimental_depth_prediction.datasets.base_contour_dataset import BaseContourAreaDataset, select_process_combinations


class SimulationDataset(BaseContourAreaDataset):
    """
    Simulated thermal sequences (FLOW-3D, Eagar-Tsai) paired with simulated melt pool masks.

    Layout:
        {thermal_root_folder}/{split}/{process condition folder}/*/{frame file}
        {labels_folder}/sdf_power_{power}_velocity_{velocity}.npy

    The label for a process condition is the middle cross-section of its mask file, and the whole file is the
    series. Masks are stored as signed distances. With sdf=False they are converted to 0/1 masks.

    Subclasses describe their file naming by implementing frame_number, parse_power_velocity,
    load_frame and experimental_key.

    Args:
        thermal_root_folder (str): Root folder of the simulated frames. The split is appended.
        labels_folder (str): Folder containing the mask files.
        start (int): Frames with a number at or below this are skipped.
        noise_factor (float): Standard deviation of the multiplicative Gaussian noise added to
            training inputs. None counts as 0.
        compress_temperatures (bool): Whether to set raw values below 2000 to 0.
        **kwargs: Passed to BaseContourAreaDataset.
    """

    # Raw simulated frames are divided by this in __getitem__.
    TEMPERATURE_SCALE = 6553.6
    # TODO: training divides simulation frames by TEMPERATURE_SCALE but the analysis path does not.
    # Check whether the analysis code expects raw values before changing this.
    ANALYSIS_TEMPERATURE_SCALE = 1

    def __init__(self, thermal_root_folder, labels_folder, start,
                 noise_factor = 0,
                 compress_temperatures = False,
                 **kwargs):
        super().__init__(thermal_root_folder = thermal_root_folder,
                         labels_folder = labels_folder,
                         **kwargs)
        self.start = start
        self.noise_factor = noise_factor or 0
        self.compress_temperatures = compress_temperatures
        self.resize = Resize(64)

        potential_folders = glob.glob(os.path.join(self.thermal_root_folder, '*'))
        self.unique_pv_combinations = sorted(os.path.basename(folder) for folder in potential_folders
                                             if 'power' in os.path.basename(folder))
        self.items = self.map_thermal_images_to_contours(
                            select_process_combinations(self.list_pvs(), fraction = self.p_v_subset, limit = 11 if self.subset else None))

    # ---- File naming, provided by each simulation source

    def frame_number(self, frame_path):
        raise NotImplementedError

    def parse_power_velocity(self, folder_name):
        raise NotImplementedError

    def load_frame(self, frame_path):
        raise NotImplementedError

    def experimental_key(self, folder_name):
        raise NotImplementedError

    # ---- Data-source methods (see BaseContourAreaDataset)

    def list_pvs(self):
        """Return the process condition folder names found under the split folder."""
        return list(self.unique_pv_combinations)

    def load_pv(self, power_velocity):
        """
        Load one process condition's frames and labels.

        Args:
            power_velocity (str): process condition folder name.

        Returns:
            tuple or None: (frames, frame_info, labels), where labels is [middle cross-section] or, with
                return_series, [middle cross-section, all masks]. None if the process condition has no mask file.
        """
        all_frame_segments, mask_images, all_frame_segment_info = self.load_thermal_images(power_velocity)
        if all_frame_segments is None:
            return None
        contour_area = mask_images[len(mask_images)//2]
        labels = [np.array(contour_area)]
        if self.return_series:
            labels.append(np.array(mask_images))
        return all_frame_segments, all_frame_segment_info, labels

    def load_thermal_images(self, power_velocity):
        """
        Load one process condition's frames and its mask file.

        Frames with a number at or below self.start are skipped. The rest are sorted by frame number.

        Args:
            power_velocity (str): process condition folder name.

        Returns:
            tuple: (mp_images, mask_images, all_frame_segment_info), or (None, None, None) if the process condition
                has no mask file.
                mp_images (numpy.ndarray): The frames, resized to 64 px if smaller and copied to 3
                    channels if self.repeat.
                mask_images (numpy.ndarray): The masks, (1, H, W).
                all_frame_segment_info (list): One dict per frame with the frame number, the experimental
                    process condition key and contour_area_length.

        Raises:
            FileNotFoundError: If labels_folder does not exist.
        """
        frames = glob.glob(os.path.join(self.thermal_root_folder, power_velocity, "*/*"))
        frames = sorted((frame for frame in frames if self.frame_number(frame) > self.start), key = self.frame_number)
        frame_numbers = np.array([self.frame_number(frame) for frame in frames]).astype('int')

        power, velocity = self.parse_power_velocity(power_velocity)
        mask_path = os.path.join(self.labels_folder, f'sdf_power_{power}_velocity_{velocity}.npy')
        try:
            # Stored upside down and with the opposite sign to the experimental SDF labels
            masks = -1*np.load(mask_path)[::-1][None, :]
        except FileNotFoundError as e:
            if not os.path.exists(self.labels_folder):
                raise FileNotFoundError(f'No label folder found at {os.path.abspath(self.labels_folder)}') from e
            print(f'No mask found for {power_velocity}')
            return None, None, None
        if not self.sdf:
            masks = (masks > 0).astype('int')

        mp_images = np.array([self.load_frame(frame) for frame in frames])
        if mp_images.shape[-1] < 64:
            mp_images = self.resize(torch.Tensor(mp_images)).numpy()
        if self.repeat:
            mp_images = np.repeat(mp_images[:, None], 3, 1)

        all_frame_segment_info = [{"frame_number": frame_number,
                                   "pv": self.experimental_key(power_velocity),
                                   "contour_area_length": len(masks)}
                                  for frame_number in frame_numbers]
        return mp_images, masks, all_frame_segment_info

    # ---- Serving samples

    def preprocess_input(self, frames):
        """Compress low temperatures, then add training noise (on raw values, before scaling)."""
        if self.compress_temperatures:
            frames[frames < 2000] = 0
        if self.split == 'train' and self.noise_factor:
            frames = frames + frames * torch.randn(frames.shape[-3:]) * self.noise_factor
        return frames

    def __getitem__(self, index):
        frames, contour, index = super().__getitem__(index)
        return frames, contour[None], index
