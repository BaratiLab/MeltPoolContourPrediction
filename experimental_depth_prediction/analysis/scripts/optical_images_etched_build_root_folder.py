# Script to build `root_folder` for torch Dataset (`cross_sections.py`).

# Requires that `data/Optical_Images_Etched/Microns/` folder contains the
# proper files.
# (i.e. run / create the necessary notebooks under `analysis/notebooks`).
# 1. Orient each of the image in the proper direction
#     * `analysis/notebooks/optical_images_oriented/`
# 2. Convert each value from pixels to microns
#     * `analysis/notebooks/optical_images_microns`

# Essentially, this script just copies files from the folders in
# `data/Optical_Images_Etched/Microns/` to one big folder in
# `data/Optical_Images_Etched/root_folder/`.

# RUN THIS FROM PROJECT ROOT DIRECTORY
# (i.e. `python ./analysis/scripts/optical_images_etched_build_root_folder.py`)

# If you update datasets, you will need to also update test results.
# `test/test_cross_sections_dataset.py`

from pathlib import Path
import glob
import shutil

csv_files = glob.glob('./data/Optical_Images_Etched/Micron/*Set*/*csv')

for filename in csv_files:
    print(f"Copying: `{filename}` to `root_folder`.")
    shutil.copy2(filename, "./data/Optical_Images_Etched/root_folder")
