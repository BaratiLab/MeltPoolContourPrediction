import datetime
import os
from os import path
from torch import nn
import matplotlib
import torch
import yaml
import wandb
from experimental_depth_prediction.runners.train import train
from experimental_depth_prediction.utils.arg_parser import parse_arguments, parse_cli_overrides
from experimental_depth_prediction.utils.utils import get_model, get_dataloader, load_model
from experimental_depth_prediction.utils.utils import plot_input_pca
matplotlib.use('Agg')

print(f'{"CUDA is available" if torch.cuda.is_available() else "CUDA is not available"}')
if not torch.cuda.is_available():
    print("No GPU available, exiting")
    exit(1)


current_directory = os.path.dirname(os.path.abspath(__file__))

args, unparsed = parse_arguments()

# Only arguments explicitly passed on the command line override the yml config
cli_args = parse_cli_overrides()


with open(os.path.join(current_directory, f'config/{args.yml}.yml'), encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)
config = yaml_config

for key, value in vars(args).items():
    if key not in config:
        print(f"Key {key} not in yaml config, using {value} as value")
        config[key] = value
for key, value in vars(cli_args).items():
    config[key] = value
config['epochs'] = int(config['epochs'])

config['batch_size'] = int(config['batch_size'])
config['weight_decay'] = float(config['weight_decay'])
config['dropout'] = float(config['dropout'])

if config['sdf']:
    config['label_file'] = '64_enlarged_sdf_contour_area'
config['seq_length'] = int(config['seq_length'])
if config['seq_length'] > 50 and 'batch_size' not in vars(cli_args):
    print(f"seq_length > 50, reducing batch_size from {config['batch_size']} to 32")
    config['batch_size'] = 32

# Select the GPU(s) before anything is moved to CUDA. CUDA_VISIBLE_DEVICES is ignored
# once CUDA has been initialized (torch.cuda.is_available() above already does this).
# Priority: --gpu N, then gpu_start/gpu_end from the config, then every visible GPU.
if config.get('gpu') is not None:
    device_ids = [config['gpu']]
elif 'gpu_start' in config and 'gpu_end' in config:
    device_ids = list(range(config['gpu_start'], config['gpu_end']))
else:
    device_ids = list(range(torch.cuda.device_count()))
torch.cuda.set_device(device_ids[0])
device = torch.device(f'cuda:{device_ids[0]}')

# Get Data
train_dataloader, test_dataloader, dev_dataloader = get_dataloader(batch_size = config['batch_size'],
                                                   config = config,
                                                   model_type = config['model_type'],
                                                   root_folder = os.path.join(current_directory, '../'),
                                                   label_file = config['label_file'],
                                                   return_dev = True)
sample_batch = next(iter(train_dataloader))
depth_img_shape = sample_batch[-2].shape[-1]

config['img_shape'] = depth_img_shape
print(len(train_dataloader), "training batches available")




if config['reload_model'] is not None:
    model = load_model(model_type = config['model_type'],
                       run_path = config['reload_model'],
                       config = config,
                       model_artifact_path=config['model_artifact_path'],
                       device = device,
                       train = True)
else:
    model = get_model(config['model_type'], config)

print(f"Now training: Parameters, Exposure: {config['exposure']}µs, " + \
      f"Image Type: {config['image_type']}, Frames: {config['seq_length']}")

# Define results folder
now = datetime.datetime.now()
datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
results_folder_suffix =  f"runs/image_{config['sequence_type']}_"+ \
                                f"augment_{config['augmentation']}area_transformer" + \
                                f"/exposure_{config['exposure']}_frames"+ \
                                f"_{config['seq_length']}" + \
                                f"/{datetime_string}"
results_folder = path.join(current_directory, '../', results_folder_suffix)
os.makedirs(results_folder, exist_ok=True)
print('Results folder:', results_folder)

# Keep the run's full settings next to its checkpoints, so runs without wandb have a record too
with open(os.path.join(results_folder, 'config.yml'), 'w', encoding='utf-8') as f:
    yaml.safe_dump(config, f, sort_keys=False)

# Only wrap in DataParallel when there is more than one GPU to split across
if config['data_parallel'] and len(device_ids) > 1:
    model = nn.DataParallel(model, device_ids = device_ids)
print(f"Training on {'GPUs ' + str(device_ids) + ' (DataParallel)' if isinstance(model, nn.DataParallel) else device}")
mode = 'online' if config.get('wandb') else 'disabled'

# Initialize wandb (disabled unless --wandb is passed)
wandb.init(
    project="ExperimentalDepthPrediction-TempTransformer",
    name=args.yml,
    entity = config.get('wandb_entity'),
    config=config,
    mode = mode
)

if config.get('plot_pca'):
    plot_input_pca(train_dataloader.dataset, 'Train Set PCA', results_folder)
    plot_input_pca(test_dataloader.dataset, 'Test Set PCA', results_folder)

# Train model
train(model = model,
      train_dataloader = train_dataloader,
      test_dataloader = test_dataloader,
      dev_dataloader= dev_dataloader,
      config_dictionary = config,
      results_folder = results_folder,
      weight_decay=config['weight_decay'],
      use_scheduler=config['use_scheduler'],
      device = device,
      method = 'SDF' if config['sdf'] else 'binary',
      resample_to_flow3d = config['resample_to_flow3d']
      )
