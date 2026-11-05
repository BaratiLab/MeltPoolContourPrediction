import glob
import json
import math
import os
from os import path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from experimental_depth_prediction.datasets.image_utils import load_image


class BaseContourAreaDataset(Dataset):
    """
    Thermal image sequences paired with melt pool cross-section labels, one label per process condition, i.e. one laser power and scan velocity pair.

    Subclasses supply the data source through two
    methods:
        list_pvs(): the process condition keys to load.
        load_pv(power_velocity): (frames, frame_info, labels) for one process condition, or None to skip it.

    For each process condition, map_thermal_images_to_contours calls load_pv and cuts the frames into windows with
    create_sequence. ContourAreaDataset implements the experimental layout and SimulationDataset the
    FLOW-3D and Eagar-Tsai layouts.

    Each entry of self.items is a tuple (window, cross_section) or, with return_series,
    (window, cross_section, series). self.items_info[i] holds the frame info dicts of the frames in
    sample i.

    Args:
        labels_folder (str): Folder containing the label map JSON files.
        exposure (str): "4" or "20" (µs), or "all" to stack both exposures as channels.
        sequence_length (int): Frames per window.
        sequence_type (str): How frames are cut into samples. "rolling" starts a window at every
            frame, "segment" uses back-to-back windows without overlap, and "mean" uses a single
            window holding all of a process condition's frames. A "_mean" suffix ("rolling_mean", "segment_mean")
            averages each window into one image. "mean" is always averaged.
        image_type (str): "Mono" or "Temp".
        thermal_root_folder (str): Root folder of the thermal images. The split is appended.
        split (str): "train", "test" or "validation". "all" uses thermal_root_folder as is.
        repeat (bool): Whether frames were copied to 3 channels for ResNet and ViT backbones. If
            False, a channel axis of size 1 is added to each window.
        model_type (str): Model type. Only used by get_pv_thermal_image.
        return_series (bool): Whether to attach each process condition's individual cross-sections (its series) to
            every sample.
        augment_transforms (bool): Whether to apply random flips and rotations to the input during
            training.
        padding (int): Half the side length of the center crop applied to every frame.
        verbose (bool): Whether to print progress details.
        moving_average_window_size (int): Frames averaged by the moving average before windowing.
            0 disables it.
        moving_average_window_stride (int): Stride of that moving average.
        randomize_contours (bool): Whether to train on a random cross-section from the series instead
            of the process condition label. Turns on return_series and needs a series file matching label_file.
        subset (bool): Whether to load only 11 process conditions, for debugging.
        label_file (str): Label map JSON to load, without the .json extension.
        p_v_subset (float, optional): Fraction of process conditions to keep, chosen at random.
        sdf (bool): Whether labels are signed distance fields. Read by the analysis code and by
            SimulationDataset.
        noise_factor (float): Not used here. See SimulationDataset.
        compress_temperatures (bool): Not used here. See SimulationDataset.
    """

    # Raw thermal frames are divided by this in __getitem__.
    TEMPERATURE_SCALE = 65536
    # Analysis reads items_pv directly instead of going through __getitem__, so
    # process_parameter_mapping divides frames by this.
    ANALYSIS_TEMPERATURE_SCALE = TEMPERATURE_SCALE

    def __init__(
        self,
        labels_folder = "data/input_label_map",
        exposure = "20",
        sequence_length = 5,
        sequence_type = "rolling",
        image_type = 'Temp',
        thermal_root_folder = "data/sequence/",
        split = "train",
        repeat = False,
        model_type = "CNN",
        return_series = False,
        augment_transforms = False,
        padding = 32,
        verbose = False,
        moving_average_window_size = 0,
        moving_average_window_stride = 1,
        randomize_contours = False,
        subset = False,
        label_file = 'contour_area',
        p_v_subset = None,
        sdf = False,
        noise_factor = 0,
        compress_temperatures = False,
    ):
        # Select the split to load here
        self.split = split
        if split == 'all':
            self.thermal_root_folder = thermal_root_folder
        else:
            self.thermal_root_folder = path.join(thermal_root_folder, self.split)
        
        self.labels_folder = labels_folder
        self.label_file = label_file
        self.image_type = image_type
        self.exposure = exposure

        # Which process conditions to load
        self.subset = subset
        if self.subset:
            print('Only using a small amount of data for debugging purposes')
        self.p_v_subset = p_v_subset

        # How frames are prepared and cut into samples
        self.padding = padding
        self.repeat = repeat
        self.moving_average_window_size = moving_average_window_size
        self.moving_average_window_stride = moving_average_window_stride
        self.sequence_length = sequence_length
        self.sequence_type = sequence_type

        # Labels
        self.randomize_contours = randomize_contours
        self.return_series = return_series or randomize_contours
        self.sdf = sdf

        # Training-time input augmentation
        self.augment_transforms = augment_transforms
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomRotation(180)])

        self.model_type = model_type
        self.verbose = verbose

        # Filled by map_thermal_images_to_contours 
        
        self.items_info = []
        self.items = []
        #  Filled by  process_parameter_mapping 
        self.items_pv = None
        self.items_info_pv = None


    # ---- Shared: turning loaded frames into samples

    def map_thermal_images_to_contours(self, process_conditions = None):
        """
        Build the samples for a list of process conditions.

        Loads each process condition with load_pv and cuts its frames into windows with create_sequence. The frame
        info of each sample is appended to self.items_info.

        Args:
            process_conditions (list, optional): process condition keys to load. Defaults to list_pvs().

        Returns:
            list: One (window, *labels) tuple per sample.
        """
        if process_conditions is None:
            process_conditions = self.list_pvs()
        
        # p_v_considered = self.list_pvs()
        # if self.p_v_subset is not None:
        #     number_considered = int(np.round(len(p_v_considered)*float(self.p_v_subset)))
        #     p_v_considered = np.random.choice(np.array(p_v_considered, dtype = 'object'), number_considered, replace = False)
        # if self.subset:  # Only use a small amount of data for debugging purposes
        #     p_v_considered = p_v_considered[:11]

        items = []
        for power_velocity in tqdm(process_conditions):
            loaded = self.load_pv(power_velocity)
            if loaded is None or len(loaded[0]) == 0:
                continue
            all_frame_segments, all_frame_segment_info, labels = loaded
            samples, samples_info = self.create_sequence(all_frame_segments, all_frame_segment_info, labels)
            items.extend(samples)
            self.items_info.extend(samples_info)
        return items

    def interpret_sequence_type(self, number_of_frames):
        """
        Translate self.sequence_type into the slicing parameters used by create_sequence.

        Args:
            number_of_frames (int): Number of frames available for the process condition.

        Returns:
            tuple: (start, stop, step, length, average). Windows start at every step-th frame from start
                up to but excluding stop, and each holds length frames. average is True when each window
                is averaged into one image.

        Raises:
            ValueError: If self.sequence_type is not rolling, segment or mean, optionally with a "_mean"
                suffix.
        """
        if self.sequence_type == 'mean':
            start, stop, step, length = 0, number_of_frames, number_of_frames, number_of_frames
        elif self.sequence_type.startswith('segment'):
            start, stop, step, length = 0, number_of_frames, self.sequence_length, self.sequence_length
        elif self.sequence_type.startswith('rolling'):
            start, stop, step, length = 0, number_of_frames - self.sequence_length + 1, 1, self.sequence_length
        else:
            raise ValueError(f"Unknown sequence_type {self.sequence_type!r}: "
                             "use rolling, segment or mean, optionally with '_mean'")
        average = self.sequence_type == 'mean' or self.sequence_type.endswith('_mean')
        return start, stop, step, length, average

    def create_sequence(self, frames, frame_info, labels):
        """
        Cut one process condition's frames into samples according to self.sequence_type.

        Args:
            frames (list): The process condition's frames in time order.
            frame_info (list): One info dict per frame.
            labels (list): Labels attached to every sample, [cross_section] or [cross_section, series].

        Returns:
            tuple: (samples, samples_info). samples holds one (window, *labels) tuple per sample. The
                window contains the frames of that window, or their mean if the sequence type averages,
                with a leading channel axis of size 1 unless self.repeat. A final window with fewer frames
                than the window length is dropped. samples_info holds the info dicts of each sample's
                frames in the same order.
        """
        start, stop, step, length, average = self.interpret_sequence_type(len(frames))
        samples, samples_info = [], []
        for i in range(start, stop, step):
            window = np.array(frames[i:i + length])
            if len(window) < length:
                break
            if average:
                window = window.mean(axis = 0)
            if not self.repeat:
                window = window[None]
            samples.append((window, *labels))
            samples_info.append(frame_info[i:i + length])
        return samples, samples_info

    def moving_average(self, values):
        """
        Smooth a sequence with a moving average.

        Each output is the mean of moving_average_window_size consecutive values, with window starts
        spaced moving_average_window_stride apart. Windows near the end run past the sequence and
        average fewer values.

        Args:
            values (list): Arrays or numbers in sequence order. Numbers are converted to float.

        Returns:
            list: The window means.
        """
        size, stride = self.moving_average_window_size, self.moving_average_window_stride
        return [np.array(values[i:i + size], dtype = 'float').mean(axis = 0)
                for i in range(0, len(values), stride)]

    # ---- Serving samples

    def preprocess_input(self, frames):
        """
        Modify the raw input tensor before augmentation and scaling.

        The base implementation returns the frames unchanged. Subclasses override this.
        """
        return frames

    def __getitem__(self, index):
        frames = self.preprocess_input(torch.Tensor(self.items[index][0]))
        if self.augment_transforms and self.split == 'train':
            frames = self.transform(frames)

        contour = self.items[index][1]
        if self.randomize_contours:
            contour_series = self.items[index][2]
            if len(contour_series) > 0:
                contour = contour_series[np.random.randint(0, len(contour_series))].reshape(contour.shape)

        return frames/self.TEMPERATURE_SCALE, contour, index

    def __getiteminfo__(self, index):
        return self.items_info[index]

    def __len__(self):
        return len(self.items)

    def get_input_label_map(self, filename):
        """
        Load a label map from the labels folder.

        Args:
            filename (str): Name of the JSON file in labels_folder, without the extension.

        Returns:
            dict: Flattened label array for each process condition key.
        """
        with open(os.path.abspath(f"{self.labels_folder}/{filename}.json"), "r", encoding = "utf-8") as fp:
            return json.load(fp)

    # ---- Per-process-condition views, used by the analysis code

    def process_parameter_mapping(self):
        """
        Group the samples by process condition for the analysis code.

        Uses the samples already in self.items, with each window divided by ANALYSIS_TEMPERATURE_SCALE.
        The groups are also stored in self.items_pv and self.items_info_pv.

        Returns:
            tuple: (items_info_pv, items_pv, items_info, items).
                items_info_pv (dict): For each process condition key, the frame info lists of its samples.
                items_pv (dict): For each process condition key, the list of its samples.
                items_info (list): The frame info list of each sample, in the order of self.items.
                items (list): Every sample, in the order of self.items.
        """
        items = [(window/self.ANALYSIS_TEMPERATURE_SCALE, *labels) for window, *labels in self.items]
        items_info = list(self.items_info)
        items_pv, items_info_pv = {}, {}
        for sample, sample_info in zip(items, items_info):
            power_velocity = sample_info[0]['pv']
            items_pv.setdefault(power_velocity, []).append(sample)
            items_info_pv.setdefault(power_velocity, []).append(sample_info)
        self.items_pv = items_pv
        self.items_info_pv = items_info_pv
        return items_info_pv, items_pv, items_info, items

    def get_pv_thermal_image(self):
        """
        Stack each process condition's input windows into one array.

        For the temporal transformer the whole window is kept. For other models only the first frame, or
        channel, of each window is kept. Calls process_parameter_mapping if it has not run yet.

        Returns:
            dict: Stacked inputs for each process condition key, with size-1 axes removed.
        """
        if self.items_pv is None:
            self.process_parameter_mapping()

        pv_thermal_image = {}
        for power_velocity, samples in self.items_pv.items():
            if self.model_type == 'temp_transformer':
                images = [window for window, *_ in samples]
            else:
                images = [window[0] for window, *_ in samples]
            pv_thermal_image[power_velocity] = squeeze_to_stack(images)
        
        return pv_thermal_image
    def pv_indices(self):
        """
        Group the sample indices of self.items by process condition.

        Returns:
            dict: List of sample indices for each process condition key.
        """
        groups = {}
        for i, sample_info in enumerate(self.items_info):
            power_velocity = sample_info[0]['pv']
            if power_velocity not in groups:
                groups[power_velocity] = []
            groups[power_velocity].append(i)
        return groups

    def get_pv_contours(self):
        """
        Stack each process condition's labels into one array.

        Each label is split along its first axis, so a (1, H, W) label contributes one (H, W)
        cross-section per sample. Calls process_parameter_mapping if it has not run yet.

        Returns:
            dict: Stacked cross-sections for each process condition key.
        """
        if self.items_pv is None:
            self.process_parameter_mapping()
        pv_contours = {}
        for power_velocity, samples in self.items_pv.items():
            cross_sections = [part for _, contour, *_ in samples for part in contour]
            pv_contours[power_velocity] = squeeze_to_stack(cross_sections)
        return pv_contours

    def list_pvs(self):
        """Return the process condition keys to load. Implemented by subclasses."""
        raise NotImplementedError

    def load_pv(self, power_velocity):
        """
        Load one process condition. Implemented by subclasses.

        Args:
            power_velocity (str): A key from list_pvs().

        Returns:
            tuple or None: (frames, frame_info, labels), or None to skip this process condition.
        """
        raise NotImplementedError

def squeeze_to_stack(arrays):
    """Stack arrays and drop size-1 axes, keeping a leading stack axis if only 2 axes are left."""
    stacked = np.squeeze(arrays)
    if stacked.ndim == 2:
        stacked = stacked[None]
    return stacked

def select_process_combinations(power_velocities, fraction = None, limit = None, rng = None):
    """
    Choose which process conditions to load.

    Args:
        power_velocities (list): The available process condition keys, in order.
        fraction (float, optional): Fraction of the keys to keep, chosen at random. None keeps all.
        limit (int, optional): Maximum number of keys to keep after the random selection. None keeps
            all.
        rng (numpy.random.Generator, optional): Random number generator for the selection. A seeded
            generator makes the choice reproducible. None uses the NumPy global random state.

    Returns:
        list: The chosen process condition keys.
    """
    chosen = list(power_velocities)
    if fraction is not None:
        number = int(np.round(len(chosen)*float(fraction)))
        choice = rng.choice if rng is not None else np.random.choice
        chosen = list(choice(np.array(chosen, dtype = 'object'), number, replace = False))
    if limit is not None:
        chosen = chosen[:limit]
    return chosen
