import json
import os

import torch
import sys
from experimental_depth_prediction.utils.utils import dict2namespace

from experimental_depth_prediction.analysis.result_analysis.analysis_tools\
      import plot_metrics, plot_train_test_metrics, plot_loss_curve, plot_metrics_model_comparison
from experimental_depth_prediction.utils.utils import create_config, get_dataloader, load_model
import argparse
import wandb
from torch import nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The wandb project main.py logs to. Override with --wandb_project.
DEFAULT_PROJECT = 'ExperimentalDepthPrediction-TempTransformer'

def end_to_end_analysis(run_path, model_artifact_path=None, show=False, create_videos = False, sweep = False, sweep_id = '', run_train_test = True, run_pv_metrics = False, 
                        plot_publication_figures = False):
    """
    Run the full result analysis for one wandb run.

    Downloads the run config, loads the model and dataloaders, and writes figures and metric
    summaries under results[_<sweep_id>]/<run id>/.

    Args:
        run_path (str): wandb run path, "<entity>/<project>/<run id>".
        model_artifact_path (str, optional): Checkpoint to load. Built from run_path if None.
        show (bool): Whether to show figures as well as saving them.
        create_videos (bool): Whether to render the frame-by-frame videos.
        sweep (bool): Whether the run belongs to a sweep. Changes the results folder.
        sweep_id (str): Sweep id used in the results folder name.
        run_train_test (bool): Whether to run the dataloader-level evaluation.
        run_pv_metrics (bool): Whether to run the per-process-condition rollout evaluation.
        plot_publication_figures (bool): Whether to also save the publication figures.
    """

    # Load in specified configuration and specify directories
    run, config, _ = create_config(run_path)
    args = dict2namespace(run.config)
    current_directory = os.path.dirname(os.path.realpath(__file__))
    if sweep:
        summary_directory = os.path.join(current_directory, '../../../', f'results_{sweep_id}', run_path.split('/')[-1], 'summary')
    else:
        summary_directory = os.path.join(current_directory, '../../../', 'results', run_path.split('/')[-1], 'summary')
 
    os.makedirs(summary_directory, exist_ok=True)


    # Load in model
    model_type = config['model_type']
    # Define parameters for batch size and sequence length
    if config['seq_length'] > 30:
        batch_size = 4
    else:
        batch_size = config['batch_size']

    print(f"Loaded: Model Type: {args.model_type}, Parameters, Exposure: {args.exposure} µs,"
            f"Image Type: {args.image_type}, Frames: {args.seq_length}, Augment: {args.augmentation},"
            f"Sequence Type: {args.sequence_type}, Sequence Folder: {args.sequence_folder}")


    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(current_folder, '../../../')
    train_dataloader, test_dataloader = get_dataloader(1, config, model_type = model_type,
                                                        root_folder = root_folder)
    img_shape = train_dataloader.dataset[0][-2].shape[-1]
 
    # breakpoint()
    print(len(train_dataloader), "training batches available")



    plot_loss_curve(run, show = show, configuration_directory = summary_directory)
    # breakpoint()

    model = load_model(model_type,
                       run_path=run_path,
                       model_artifact_path=model_artifact_path,
                       config=config,
                       device=device)
    if run_train_test:
        plot_train_test_metrics(
            model,
            train_dataloader,
            test_dataloader,
            run_path=run_path,
            show=show,
            img_shape=img_shape,
            sweep_id=sweep_id,
            config=config,
            plot_publication_figures = plot_publication_figures

        )
    if run_pv_metrics:
        error_metrics = {}
        partition = 'test'
        for partition in ['test', 'train']:
            if partition == 'test':
                error_metrics[partition] = plot_metrics(test_dataloader.dataset, 
                                                        partition, 
                                                        model = model,
                                                        model_type = model_type,  
                                                        debug_plot= False, 
                                                        run_path = run_path, 
                                                        show = show,
                                                        create_videos = create_videos,
                                                        img_shape = img_shape, 
                                                        sweep_id = sweep_id)
            elif partition == 'train':
                error_metrics[partition] = plot_metrics(train_dataloader.dataset, 
                                                        partition, 
                                                        model = model,
                                                        model_type = model_type,  
                                                        debug_plot = False, 
                                                        run_path = run_path, 
                                                        show = show,
                                                        create_videos = create_videos,
                                                        img_shape = img_shape,
                                                        sweep_id = sweep_id)

        with open(os.path.join(summary_directory, 'rollout_error_metrics.json'), 'w') as f:
                json.dump(error_metrics, f)


def analyze_sweep(sweep_id, entity, project = DEFAULT_PROJECT, create_videos=False, redo_completed_runs = False, run_train_test = False, run_pv_metrics = False, plot_publication_figures = False):
    """
    Run end_to_end_analysis on every run of a wandb sweep.

    Args:
        sweep_id (str): Sweep id under entity/project.
        entity (str): wandb user or team the sweep belongs to.
        project (str): wandb project the sweep belongs to.
        create_videos (bool): Passed to end_to_end_analysis.
        redo_completed_runs (bool): Whether to analyze runs that already have results.
        run_train_test, run_pv_metrics, plot_publication_figures (bool): Passed to
            end_to_end_analysis.
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    for run in sweep.runs:
        run_path = os.sep.join(run.path)
        run, config, config_directory = create_config(run_path)
        json.dump(run.config, open(os.path.join(f"{config_directory}", "config.json"), 'w' ) )
        model_type = config['model_type']
        
        current_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(current_folder, '../../../')
        if not redo_completed_runs:
            if os.path.exists(f'{root_folder}/results_{sweep_id}/{run.id}/summary/error_metrics.json'):
                print('Skipping', run_path, f'Image Type: {config["image_type"]}, Model Type: {model_type}, already processed')
                continue

        run_id = run.id
        model_artifact_path = f'{entity}/{project}/{model_type}-{run_id}:v2'
       
        if not 'pretrain_flow3d' in config:
            config['pretrain_flow3d'] = False
        
        print('\033[91m' + f"Now running {sweep_id}:{run_id}" + '\033[0m')
        try:
            end_to_end_analysis(run_path=run_path, show=False, model_artifact_path=model_artifact_path, create_videos=create_videos, sweep = True, 
                                sweep_id = sweep_id, run_train_test = run_train_test, run_pv_metrics = run_pv_metrics, plot_publication_figures = plot_publication_figures)
        except wandb.errors.CommError as e:
            print(f"Wandb error occured on {sweep_id}:{run_id}")
            print(e)
            continue
def main():

   
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--run_all', action='store_true', default=False, help="Run all experiments")
    parser.add_argument('-e', '--experiment', type = str, help="Experiment ID") 
    parser.add_argument('--batch_size', type = int, default = 1, help="Batch Size")
    parser.add_argument('-v', '--version', type = int, default = 0, help="Version of model")
    parser.add_argument('-c', '--create_videos', action = 'store_true', default = False, help="Create videos?")
    parser.add_argument('-p', '--path',type = str, default = None, help="Path to saved model")
    parser.add_argument('-s', '--sweep', type = str, default = None, help = "ID of sweep to run")
    parser.add_argument('-t', '--train_test', action = 'store_true', default = False, help = "Run train test metrics")
    parser.add_argument('-pv', '--pv_metrics', action = 'store_true', default = False, help = "Run pv metrics")
    parser.add_argument('-rc', '--redo_completed_runs', action = 'store_true', default = False, help = "Redo completed runs")
    parser.add_argument('-ps', '--plot_publication_figures', action = 'store_true', default = False, help = "Making publication style plots of the melt pool?")
    parser.add_argument('--wandb_entity', type = str, default = None, help = "wandb user or team the runs are under (default: your wandb default)")
    parser.add_argument('--wandb_project', type = str, default = DEFAULT_PROJECT, help = "wandb project the runs are in")
    args = parser.parse_args()
    # A run path needs an actual entity name, so fall back to the logged-in user's default
    entity = args.wandb_entity or wandb.Api().default_entity
    project = args.wandb_project
    if args.sweep is not None:
        analyze_sweep(
            args.sweep,
            entity,
            project,
            create_videos=args.create_videos,
            redo_completed_runs=args.redo_completed_runs,
            run_train_test=args.train_test,
            run_pv_metrics=args.pv_metrics,
            plot_publication_figures = args.plot_publication_figures
        )
    else:
        if args.run_all:
            # put your own personal run paths here
            run_paths = [
                # example: 'fogoke/ExperimentalDepthPrediction-TempTransformer/06phgzi4',
            ]
            model_artifact_paths = [None, None, None]
            for run_path, model_artifact_path in zip(run_paths, model_artifact_paths):
                end_to_end_analysis(
                    run_path=run_path,
                    model_artifact_path=model_artifact_path,
                    show=False,
                plot_publication_figures = args.plot_publication_figures
                )
        else:
            run_path = f'{entity}/{project}/{args.experiment}'

            run, config, config_directory = create_config(run_path)

            if not 'pretrain_flow3d' in config:
                config['pretrain_flow3d'] = False
            model_type = config['model_type']
            if args.path is None:
                model_artifact_path = f'{entity}/{project}/{model_type}-{args.experiment}:v{args.version}'
            else:
                model_artifact_path = args.path
            
            end_to_end_analysis(
                run_path=run_path,
                show=False,
                model_artifact_path=model_artifact_path,
                create_videos=args.create_videos,
                run_train_test=args.train_test,
                run_pv_metrics=args.pv_metrics,
                plot_publication_figures = args.plot_publication_figures
            )



if __name__ == '__main__':
    main()