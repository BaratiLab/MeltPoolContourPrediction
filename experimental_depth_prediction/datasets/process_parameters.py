import glob
import json
import math
import numpy as np
import torch

from torch.utils.data import Dataset
from os import listdir, path
from tqdm import tqdm 

# Uses directory name relative to this file
# (Useful for imports if called outside of folder.)
dirname = path.dirname(__file__)

class ProcessParamDataset(Dataset):
    def __init__(
        self,
        contour_parameter_maps_root_folder = path.join(dirname, "./contour_parameter_maps"),
        exposure = "20", # "20" or "4"
        polynomial_order = 4,
        sequence_length = 5,
        sequence_type = "rolling", # "segment" or "rolling"
        image_type = 'Temp', # "Mono" or "Temp"
        thermal_root_folder = path.join(dirname, "sequence/"),
        verbose = False, # used for debugging
        split="train"
    ):
        self.contour_parameter_maps_root_folder = contour_parameter_maps_root_folder
        self.exposure = exposure
        self.polynomial_order = polynomial_order
        self.sequence_length = sequence_length
        self.sequence_type = sequence_type
        self.split = split
        self.thermal_root_folder = path.join(thermal_root_folder, self.split)
        self.verbose = verbose
        self.image_type = image_type
        self.contour_parameter_map = self.get_contour_parameter_map()
        self.items_info = []
        self.items = self.map_thermals_to_contours()

    def __getitem__(self, index):
        return self.items[index][0], self.items[index][1]

    def __getiteminfo__(self, index):
        return self.items_info[index]

    def testgetitem(self, index):
        return self.items[index][0], self.items[index][1]

    def __len__(self):
        return self.dataset_length

    def get_contour_parameter_map(self):
        print(path.abspath(f"{self.contour_parameter_maps_root_folder}/all.json"))
        with open(f"{self.contour_parameter_maps_root_folder}/all.json", "r") as fp:
            return json.load(fp)

    def map_thermals_to_contours(self):
        """
        Maps each thermal image frame (i.e. 1.npy) to the possible contours.
        `items` would be something like this:
        [
            ([1.npy, 2.npy, 3.npy, 4.npy, 5.npy], [coeffcients_set_1]),
            ([1.npy, 2.npy, 3.npy, 4.npy, 5.npy], [coeffcients_set_2]),
            ([1.npy, 2.npy, 3.npy, 4.npy, 5.npy], [coeffcients_set_3]),
            ([1.npy, 2.npy, 3.npy, 4.npy, 5.npy], [coeffcients_set_4]),
        ]
        """
        items = []

        X = 50
        Y = 100
        p = 32 # padding
        if self.verbose:
            print(f"Loading folders...")
        for power_velocity in tqdm(listdir(self.thermal_root_folder)):
            folder =  f"{self.thermal_root_folder}/{power_velocity}"
         
            if self.verbose:
                print(f"Folder found: {folder}")
            contours = self.contour_parameter_map[f"order_{self.polynomial_order}"][power_velocity]
            frames = sorted(glob.glob(f"{folder}/{self.image_type}/{self.exposure}/*.npy"), key=len)
          
            # Number of non-overlapping segments in a total set of frames.
            segments = math.floor(len(frames) / self.sequence_length)
       
            # Number of windows with stride of 1.
            windows = len(frames) - self.sequence_length + 1

            for contour_index, contour in enumerate(contours):

                if self.sequence_type == "segment":
                    for segment in range(segments):
                        frames_segment = []
                        frames_segment_info = []
                        for sequence_index in range(self.sequence_length):
                            frame_index = (segment * self.sequence_length) + sequence_index
                            image_npy = np.load(frames[frame_index])[X-p:X+p,Y-p:Y+p]/65536
                            frames_segment.append(image_npy.astype(float))
                            frames_segment_info.append({
                                "frame_number": frame_index + 1,
                                "pv": power_velocity,
                                "contour_index": contour_index,
                                "contour_count": len(contours)
                            })
                        self.items_info.append(frames_segment_info)
                        pv = np.array(frames_segment_info[-1]['pv'].split('_')).astype(float)
                        frame_window_arr = np.array(frames_window)
                        frame_window_info = np.array(frames_window_info[-1]['pv'].split('_')).astype(float)
                        normalized_frame_window_info = frame_window_info/np.array([400, 2.4])
                        items.append(frame_window_arr, normalized_frame_window_info)
                        # items.append((np.array(frames_segment), np.array(frames_segment_info[-1]['pv'].split('_')).astype(float)/np.array([400, 2.4])))



                elif self.sequence_type == "rolling":
                    for window in range(windows):
                        frames_window = []
                        frames_window_info = []
                        for sequence_index in range(self.sequence_length):
                            frame_index = window + sequence_index - 1
                            image_npy = np.load(frames[frame_index])[X-p:X+p,Y-p:Y+p]/65536
                            frames_window.append(image_npy.astype(float))
                            frames_window_info.append({
                                "frame_number": frame_index + 1,
                                "pv": power_velocity,
                                "contour_index": contour_index,
                                "contour_count": len(contours)
                            })
                        self.items_info.append(frames_window_info)
                        frame_window_arr = np.array(frames_window)
                        frame_window_info = np.array(frames_window_info[-1]['pv'].split('_')).astype(float)
                        normalized_frame_window_info = frame_window_info/np.array([400, 2.4])
                        items.append(frame_window_arr, normalized_frame_window_info)
                        # items.append((np.array(frames_window), np.array(frames_window_info[-1]['pv'].split('_')).astype(float)/np.array([400, 2.4])))


        self.dataset_length = len(items)
        return items
