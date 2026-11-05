import glob
from torchvision.transforms import Resize
from os import path
import os
import numpy as np
import math
from experimental_depth_prediction.datasets.base_contour_dataset import BaseContourAreaDataset, select_process_combinations
from experimental_depth_prediction.datasets.image_utils import load_image
class ContourAreaDataset(BaseContourAreaDataset):
    """
    Experimental thermal sequences paired with the melt pool cross-section label of each process condition.

    Frames are read from
        {thermal_root_folder}/{split}/{power}_{velocity}/{image_type}/{exposure}/{frame}.npy
    Labels come from the JSON label maps in labels_folder. label_file holds the training label of
    each process condition, and "128_enlarged_max_contour_area_series" holds the per-condition series
    used with return_series. "original" is also loaded.

    Takes the arguments of BaseContourAreaDataset.
    """
    def __init__(self, *args, **kwargs):
        super(ContourAreaDataset, self).__init__(*args, **kwargs)
        self.labels_contour_area = self.get_input_label_map(self.label_file)
        self.labels_original = self.get_input_label_map("original")
        self.labels_contour_series = self.get_input_label_map("128_enlarged_max_contour_area_series")
       
        # self.augment_labels = self.augment_labels
        self.items_info = []
        potential_folders = glob.glob(os.path.join(self.thermal_root_folder, self.split, '*'))

        self.unique_pv_combinations = sorted([folder.split('/')[-1] for folder in potential_folders if 'power' in folder])

        self.items = self.map_thermal_images_to_contours(
                            select_process_combinations(self.list_pvs(), fraction = self.p_v_subset, limit = 11 if self.subset else None))

        self.resize = Resize(224)

    # ---- Data-source methods (see BaseContourAreaDataset)

    def list_pvs(self):
        """Return every process condition key in the label map."""
        return list(self.labels_contour_area.keys())

    def load_pv(self, power_velocity):
        """
        Load one process condition's frames and labels.

        Args:
            power_velocity (str): Key in the label map, e.g. "280_0.45".

        Returns:
            tuple: (frames, frame_info, labels), where labels is [contour_area] or, with return_series,
                [contour_area, contour_series]. contour_area is reshaped to (1, H, W).
        """
        # Labels are stored flattened and assumed to be square images
        contour_area = np.array(self.labels_contour_area[power_velocity])
        img_side_length = int(np.sqrt(contour_area.size))
        contour_area = contour_area.reshape((1, img_side_length, img_side_length))
        labels = [contour_area]
        
        if self.return_series:
            labels.append(np.array(self.labels_contour_series[power_velocity]))

        all_frame_segments, all_frame_segment_info = self.load_thermal_images(power_velocity)
        return all_frame_segments, all_frame_segment_info, labels

    def load_thermal_images(self, power_velocity):
        """
        Load, crop and smooth one P/V's thermal frames.

        Reads {thermal_root_folder}/{power_velocity}/{image_type}/{exposure}/{frame}.npy, in frame
        order. With exposure "all", only frames present in both the 4 and 20 µs folders are used,
        stacked as 2 channels.

        P/Vs with fewer than 2 * sequence_length frames have their frames repeated end to end until
        they have enough. Windows can then run from the last frame back to the first.

        Returns:
            all_frame_segments (list): the frames (raw values, not divided by TEMPERATURE_SCALE),
                after the moving average if moving_average_window_size > 0.
            all_frame_segment_info (list): one dict per frame: frame number, P/V and
                contour_area_length (always 1: one label per P/V).
        """
        folder = f"{self.thermal_root_folder}/{power_velocity}/{self.image_type}"
        if self.verbose:
            print(f"Template folder: {folder}/{self.exposure}/*.npy")

        def list_frame_numbers(exposure):
            return [os.path.basename(frame).split('.')[0] for frame in glob.glob(f"{folder}/{exposure}/*.npy")]

        def load_frames(exposure):
            return [load_image(f"{folder}/{exposure}/{frame_number}.npy", repeat = self.repeat, padding = self.padding).astype('float')
                    for frame_number in frame_numbers]

        if self.exposure == 'all':
            frame_numbers = list(set(list_frame_numbers(4)) & set(list_frame_numbers(20)))
        else:
            frame_numbers = list_frame_numbers(self.exposure)
        frame_numbers.sort(key = int)

        # Repeat short sequences so there are at least 2*sequence_length frames
        if 0 < len(frame_numbers) < 2*self.sequence_length:
            frame_numbers = frame_numbers * math.ceil(2*self.sequence_length / len(frame_numbers))

        if self.exposure == 'all':
            all_frame_segments = [np.concatenate((frame_4[None], frame_20[None]), axis = 0)
                                  for frame_4, frame_20 in zip(load_frames(4), load_frames(20))]
        else:
            all_frame_segments = load_frames(self.exposure)

        if self.moving_average_window_size > 0:
            all_frame_segments = self.moving_average(all_frame_segments)
            frame_numbers = self.moving_average(frame_numbers)

        all_frame_segment_info = [{"frame_number": frame_number,
                                   "pv": power_velocity,
                                   "contour_area_length": 1}
                                  for frame_number in frame_numbers]
        return all_frame_segments, all_frame_segment_info


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_root_folder = path.join(current_folder, '../../data/')

    dataset = ContourAreaDataset(sequence_type = 'segment_mean', 
                                 padding = 32, 
                                 split = '',
                                 sequence_length = 10, 
                                 exposure = '4', 
                                 labels_folder = os.path.join(dataset_root_folder, 'input_label_map'),
                                 thermal_root_folder = os.path.join(dataset_root_folder, 'thermal'), 
                                 moving_average_window_size=10,
                                 p_v_subset = 0.80)
    


    print(len(dataset))
    sample = dataset[0]
    print(sample[0].shape)
    dataset.process_parameter_mapping()

    

if __name__ == "__main__":
    main()
