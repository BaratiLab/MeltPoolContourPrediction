import glob
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import os
from torchvision.transforms import Resize 
from experimental_depth_prediction.datasets.simulation_dataset import SimulationDataset
from experimental_depth_prediction.utils.data_utils import convert_ET_to_experimental_pv


class ETDataset(SimulationDataset):
    def __init__(self,
                labels_folder = "data/ET_sequence/depths_full_smoothed",
                thermal_root_folder = "data/ET_sequence/temp_full_smoothed",
                start = 0,
                **kwargs):


        super().__init__(thermal_root_folder = thermal_root_folder,
                labels_folder = labels_folder,
                start = start,
                **kwargs)

        
    def frame_number(self, frame_path):          # .../temp_45.npy
        return int(frame_path.split('temp_')[-1].split('.npy')[0])

    def parse_power_velocity(self, folder_name):  # power_280_velocity450
        power = folder_name.split('power_')[-1].split('_')[0]
        velocity = folder_name.split('_')[-1].split('velocity')[-1]
        return power, velocity

    def load_frame(self, frame_path):
        return np.load(frame_path)

    def experimental_key(self, folder_name):
        return convert_ET_to_experimental_pv(folder_name)
