import glob
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import os
from torchvision.transforms import Resize 
from experimental_depth_prediction.utils.data_utils import convert_flow3d_to_experimental_pv
from experimental_depth_prediction.datasets.simulation_dataset  import SimulationDataset
from experimental_depth_prediction.datasets.contour_area_image import ContourAreaDataset

class Flow3DDataset(SimulationDataset):
  

    def __init__(
        self,
    labels_folder = "data/Flow3D/masks_full_smoothed",
    thermal_root_folder = "data/Flow3D/temp_full_smoothed",
        start = 40, **kwargs):


        super().__init__(thermal_root_folder = thermal_root_folder,
                        labels_folder = labels_folder,
                        start = start,
                        **kwargs)

    def frame_number(self, frame_path):          # .../temp45
        return int(frame_path.split('temp')[-1])

    def parse_power_velocity(self, folder_name):  # power280_velocity450
        power = folder_name.split('_')[0].split('power')[-1]
        velocity = folder_name.split('_')[1].split('velocity')[-1]
        return power, velocity

    def load_frame(self, frame_path):
        return np.loadtxt(frame_path)

    def experimental_key(self, folder_name):
        return convert_flow3d_to_experimental_pv(folder_name)
       