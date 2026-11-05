import argparse 

def str2bool(value):
    """
    Convert a command line string such as 'True'/'false'/'1'/'no' to a bool.
    argparse's type=str2bool treats any non-empty string (including 'False') as True.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}')

def get_parser():
    """
    Build the command line parser for the Experimental Depth Prediction script.
    
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('-y', '--yml', type=str, default = 'finalized_temp_transformer', help='YML file to use for configuration')
    parser.add_argument('--seq_length', type=int, help='Number of thermal image frames to use for model training')
    parser.add_argument('--exposure', type=str, help='Exposure time for dataset')
    parser.add_argument('--model_type', type=str, help='Model type to use (CNN, CNNLSTM, ViT)')
    parser.add_argument('--batch_size', type=int, help='Model batch size')
    parser.add_argument('--augmentation', type=str2bool, help='Augment Data? (T/F)')
    parser.add_argument('--gpu', type=int, help='GPU ID to use for training')
    parser.add_argument('--padding', type=int, help='Amount of padding to use for training')
    parser.add_argument('--pretraining', type=str2bool, help='Use pretrained model? (T/F)')
    parser.add_argument('--sequence_folder', type=str, help='Which train/test split to use (sequence/sequence_exposure)')
    parser.add_argument('--sequence_type', type=str, help='Which sequence type to use (segment/rolling/segment_mean)')
    parser.add_argument('--image_type', type=str, help='Temp -> use temperature, Mono -> use monochrome images')
    parser.add_argument('--embed_only', type=str2bool, help='True -> only use embedding, False -> use embedding and MLP')
    parser.add_argument('--augment_transforms', type=str2bool, help='True -> use augmentations, False -> no augmentations')
    parser.add_argument('--moving_average_window_size', type=int, help='Size of moving window for temporal smoothing')
    parser.add_argument('--moving_average_window_stride', type=int, help='Stride of moving window for temporal smoothing')
    parser.add_argument('--transformer_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--dim', type=int, help='Dimension of transformer')
    parser.add_argument('--wandb_entity', type=str, help='wandb user or team to log runs under (default: your wandb default)')
    parser.add_argument('--wandb', action='store_true', help='Log the run to Weights & Biases (off by default)')

    parser.add_argument('--num_heads', type=int, help='Number of heads in transformer')
    parser.add_argument('--ff_dim', type=int, help='Feed forward dimension in transformer')
    parser.add_argument('--dropout', type=float, help='Dropout in transformer')
    parser.add_argument('--learning_rate', '-lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs')
    parser.add_argument('--data_parallel', type=str2bool, help='Use data parallel?')
    parser.add_argument('--backbone', type=str, help='Backbone for temp transformer')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--use_scheduler', action='store_true', help='Use scheduler?')
    parser.add_argument('--randomize_contours', action='store_true', help='Randomize contours?')
    parser.add_argument('--pretrain_flow3d', action='store_true', help='Pretrain with flow3d?')
    parser.add_argument('--model_log_frequency', default = 1, type=int, help='Frequency to log model')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode?')
    parser.add_argument('--subset', action = 'store_true', help='If true, only use a small amount of data for training')
    parser.add_argument('--reload_model', type = str, help='Reload model from specified path')
    parser.add_argument('--model_artifact_path', type = str, help='Model artifact path')
    parser.add_argument('--label_file', type = str, help='Label file to use for training')
    parser.add_argument('--sdf', action='store_true', help='Use SDF?')
    parser.add_argument('--dataset_ratio', type = float, help='Portion of the dataset to use')
    parser.add_argument('--resample_to_flow3d', action='store_true', help='Resample to flow3d?')
    parser.add_argument('--compress_temperatures', action='store_true', help='Compress to temperatures?')
    parser.add_argument('--noise_factor', type=float, help='Amount of noise to apply in augmentation.')
    return parser

def parse_arguments():
    """
    Parse command line arguments for the Experimental Depth Prediction script.
    
    Returns:
        tuple: Parsed command line arguments (argparse.Namespace) and the list of unparsed arguments.
    """
    return get_parser().parse_known_args()

def parse_cli_overrides():
    """
    Parse only the arguments that were explicitly passed on the command line, with
    their proper types. Arguments that were not passed are left out of the namespace,
    so they do not override values from the yml config.

    Returns:
        argparse.Namespace: The explicitly passed command line arguments.
    """
    parser = get_parser()
    for action in parser._actions:
        if action.dest != 'help':
            action.default = argparse.SUPPRESS
    cli_args, _ = parser.parse_known_args()
    return cli_args