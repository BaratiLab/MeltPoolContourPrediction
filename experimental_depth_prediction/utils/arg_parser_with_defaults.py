import argparse 

def parse_arguments():
    """
    Parse command line arguments for the Experimental Depth Prediction script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('-y', '--yml', type=str, default='default', help='YML file to use for configuration')
    parser.add_argument('--seq_length', type=int, default="15", help='Number of thermal image frames to use for model training')
    parser.add_argument('--exposure', type=str, default="20", help='Exposure time for dataset')
    parser.add_argument('--model_type', type=str, default='temp_transformer', help='Model type to use (CNN, CNNLSTM, ViT)')
    parser.add_argument('--batch_size', type=int, default=32, help='Model batch size')
    parser.add_argument('--augmentation', type=bool, default=True, help='Augment Data? (T/F)')
    parser.add_argument('--gpu', type=int, default=6, help='GPU ID to use for training')
    parser.add_argument('--padding', type=int, default=16, help='Amount of padding to use for training')
    parser.add_argument('--pretraining', type=bool, default=False, help='Use pretrained model? (T/F)')
    parser.add_argument('--sequence_folder', type=str, default='sequence_exposure', help='Which train/test split to use (sequence/sequence_exposure)')
    parser.add_argument('--sequence_type', type=str, default='rolling', help='Which sequence type to use (segment/rolling/segment_mean)')
    parser.add_argument('--image_type', type=str, default='Temp', help='Temp -> use temperature, Mono -> use monochrome images')
    parser.add_argument('--embed_only', type=bool, default=False, help='True -> only use embedding, False -> use embedding and MLP')
    parser.add_argument('--augment_transforms', type=bool, default=False, help='True -> use augmentations, False -> no augmentations')
    parser.add_argument('--moving_average_window_size', type=int, default=5, help='Size of moving window for temporal smoothing')
    parser.add_argument('--moving_average_window_stride', type=int, default=1, help='Stride of moving window for temporal smoothing')
    parser.add_argument('--transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dim', type=int, default=256, help='Dimension of transformer')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in transformer')
    parser.add_argument('--ff_dim', type=int, default=1024, help='Feed forward dimension in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout in transformer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--data_parallel', type=bool, default=False, help='Use data parallel?')
    parser.add_argument('--backbone', type=str, default='CNN', help='Backbone for temp transformer')
    parser.add_argument('--weight_decay', type=float, default=1E-4, help='Weight decay for optimizer')
    parser.add_argument('--use_scheduler', type=bool, default=True, help='Use scheduler?')
    parser.add_argument('--dice_loss', action='store_true', help='Use dice loss?')
    parser.add_argument('--area_loss', action='store_true', help='Use area loss?')
    parser.add_argument('--randomize_contours', action='store_true', help='Randomize contours?')
    parser.add_argument('--pretrain_flow3d', action='store_true', help='Pretrain with flow3d?')
    parser.add_argument('--model_log_frequency', type=int, default=1, help='Frequency to log model')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode?')
    parser.add_argument('--label_file', type = str, help='Label file to use for training')
    parser.add_argument('--sdf', action='store_true', help='Use SDF?')
    parser.add_argument('--dataset_ratio', type = float, default = 1, help='Portion of the dataset to use')
    parser.add_argument('--resample_to_flow3d', action='store_true', help='Resample to flow3d?')
    parser.add_argument('--compress_temperatures', action='store_true', help='Compress to temperatures?')
    parser.add_argument('--noise_factor', type=float,default = 0, help='Amount of noise to apply in augmentation.')
    return parser.parse_args()