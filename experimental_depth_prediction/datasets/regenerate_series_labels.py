"""
Regenerate data/input_label_map/128_enlarged_{max,mean}_contour_area_series.json.

These files are not in git. ContourAreaDataset loads 128_enlarged_max_contour_area_series when it
is constructed. They were originally produced by cell 24 of
experimental_depth_prediction/analysis/notebooks/archive/v3/contour_area.ipynb. This script runs
only the cells that build them (1-6, 22 and 24), from the contour CSVs in
data/Optical_Images_Etched/root_folder.

    python experimental_depth_prediction/datasets/regenerate_series_labels.py
    python experimental_depth_prediction/datasets/regenerate_series_labels.py --keep-first-contour
    python experimental_depth_prediction/datasets/regenerate_series_labels.py --output-root /tmp/out

By default the notebook is reproduced exactly. That drops the first contour seen for each process
condition from the series (71 of 111 contours kept, 5 keys left empty), and which contour is
dropped depends on glob order, so the output differs between machines. --keep-first-contour keeps
all 111 contours and the output is then the same on every machine apart from the order of
contours within each key.

Requires numpy, matplotlib, scikit-image, scipy and tqdm.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NOTEBOOK = os.path.join(REPO_ROOT, 'experimental_depth_prediction', 'analysis', 'notebooks',
                        'archive', 'v3', 'contour_area.ipynb')
CELLS = [1, 2, 3, 4, 5, 6, 22, 24]

# The notebook's paths assume it runs from analysis/notebooks/v3, four levels below the repo root
# (it now sits one level deeper, in archive/v3).
NOTEBOOK_CSV_PATH = '"../../../../datasets/Optical_Images_Etched'
NOTEBOOK_OUTPUT_ROOT = "os.path.abspath(os.path.join(os.getcwd(), '../../../../'))"
# The notebook writes under datasets/, which has since been renamed to data/
NOTEBOOK_LABEL_FOLDER = './datasets/input_label_map/'
# Cell 6 returns (values, parameters, micron_values), but cell 22 still unpacks two values.
# micron_values is just the last CSV's raw data and is not used to build the labels.
NOTEBOOK_UNPACK = 'contour_normalized_values, contour_parameters = normalize_and_define_contour_parameters('
FIXED_UNPACK = 'contour_normalized_values, contour_parameters, _ = normalize_and_define_contour_parameters('
# Cell 24: the first contour for each power/velocity creates an empty list instead of being stored
DROPPED_FIRST_CONTOUR = 'contour_parameter_map[f"{power}_{velocity}"] = []'
KEPT_FIRST_CONTOUR = 'contour_parameter_map[f"{power}_{velocity}"] = [list(downsampled_mask.flatten().astype(float))]'


def load_cell(notebook, index, output_root, keep_first_contour):
    source = ''.join(notebook['cells'][index]['source'])
    source = source.replace(NOTEBOOK_CSV_PATH, f'"{REPO_ROOT}/data/Optical_Images_Etched')
    source = source.replace(NOTEBOOK_OUTPUT_ROOT, repr(output_root))
    source = source.replace(NOTEBOOK_LABEL_FOLDER, './data/input_label_map/')
    source = source.replace(NOTEBOOK_UNPACK, FIXED_UNPACK)
    if keep_first_contour and index == 24:
        if source.count(DROPPED_FIRST_CONTOUR) != 1:
            raise RuntimeError('Cell 24 changed; cannot apply --keep-first-contour')
        source = source.replace(DROPPED_FIRST_CONTOUR, KEPT_FIRST_CONTOUR)
    return source


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', default = REPO_ROOT,
                        help = 'Files are written to <output-root>/data/input_label_map/ (default: the repo)')
    parser.add_argument('--keep-first-contour', action = 'store_true',
                        help = 'Keep the first contour for each power/velocity (fixes the notebook bug)')
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    output_folder = os.path.join(output_root, 'data', 'input_label_map')
    os.makedirs(output_folder, exist_ok = True)
    with open(NOTEBOOK, encoding = 'utf-8') as f:
        notebook = json.load(f)

    scope = {}
    for index in CELLS:
        source = load_cell(notebook, index, output_root, args.keep_first_contour)
        exec(compile(source, f'contour_area.ipynb[cell {index}]', 'exec'), scope)

    for method in ['max', 'mean']:
        path = os.path.join(output_folder, f'128_enlarged_{method}_contour_area_series.json')
        with open(path, encoding = 'utf-8') as f:
            series = json.load(f)
        lengths = [len(contours) for contours in series.values()]
        print(f'{path}: {len(series)} keys, {sum(lengths)} contours, '
              f'{sum(n == 0 for n in lengths)} empty keys')


if __name__ == '__main__':
    main()
