import os
from collections import OrderedDict
import json
from pathlib import Path
from datetime import datetime
import argparse
import torch
import numpy as np
import sys


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Test a UNet Model")

    # --- Define all your arguments here ---
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, default="config/SpecUNet.json", help="Config file path")

    # Reproducibility
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on",
    )

    # Train or test
    parser.add_argument("--train", action="store_true", default=None, help="Train the model")
    parser.add_argument("--test_sim", action="store_true", default=None, help="Test simulated data on model")
    parser.add_argument("--test_exp", action="store_true", default=None,help="Test experimental data on model")
    parser.add_argument("--train_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--train_size", type=int, default=0, help="Training data size")
    parser.add_argument("--test_path", type=str, default=None, help="Path to testing data")
    parser.add_argument("--test_size", type=int, default=0, help="Test data size")

    # Dataset
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--norm", action="store_true", default=None, help="Is data normalized?")
    parser.add_argument("--input_size", type=tuple, help="Input image size")
    parser.add_argument("--validation_split", type=float, help="Validation split (if validation is required)")
    parser.add_argument("--num_workers", type=int, help="Number of workers")
    parser.add_argument("--input_name", type=str, default=None, help="Raw spectral image")
    parser.add_argument("--target_name", type=str, default=None, help="Ground truth background image")
    parser.add_argument("--GTspt", type=str, default=None, help="Ground truth spectral image")
    parser.add_argument("--spt", type=str, default=None, help="Spectral info")

    # Model training
    parser.add_argument("--model_type", choices=["unet", "all"], type=str,
                        help="Model type (unet, all)")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="L2-regularization factor")
    parser.add_argument("--loss_fn", choices=["mse", "mae", "huber", "combined", "rmse"], type=str)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], type=str)
    parser.add_argument("--initializer", choices=["kaiming_normal", "xavier_uniform"], type=str)
    # LR scheduler
    parser.add_argument("--lr_scheduler", choices=["step", "cosine", "cyclic", "one_cycle"], type=str)
    parser.add_argument("--scheduler_step_size", type=int)
    parser.add_argument("--scheduler_gamma", type=float)

    args = parser.parse_args()

    # TODO: override json object if any of train, test_sim, test_exp is specified

    # --- Validation checks ---
    # if not args.train and not args.test_sim and not args.test_exp:
    #     parser.error("No action specified. Please add --train or --test_sim or --test_exp")

    if args.test_sim and args.test_exp:
        parser.error("Cannot test both simulated and experimental data at once")

    if args.train:
        # if args.lr is None or args.weight_decay is None:
        #     parser.error("Both --lr and --weight_decay are required for training")

        if args.validation_split is not None:
            if args.validation_split < 0 or args.validation_split > 1:
                parser.error("Train and test split must be between 0 and 1")

            if args.validation_split == 0 and args.test_size == 0:
                parser.error("No validation data provided during training")


    return args


def parse_json(args):
    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    ''' replace the config context using args '''
    if args.exp_name is not None:
        opt['experiment_name'] = args.exp_name

    # Override phase with args
    if args.train or args.test_sim or args.test_exp:
        if args.train and opt['phase']['train'] != args.train:
            opt['phase']['train'] = args.train
        if args.test_sim and opt['phase']['test_sim'] != args.test_sim:
            opt['phase']['test_sim'] = args.test_sim
        elif args.test_exp and opt['phase']['test_exp'] != args.test_exp:
            opt['phase']['test_exp'] = args.test_exp

    if args.device is not None:
        opt['device_args']['device'] = args.device

    if args.seed is not None:
        opt['datasets']['data_type']['seed'] = args.seed

    if args.norm is not None:
        opt['datasets']['data_type']['norm'] = args.norm

    if args.input_name is not None:
        opt['datasets']['data_type']['input_name'] = args.input_name

    if args.target_name is not None:
        opt['datasets']['data_type']['target_name'] = args.target_name

    if args.GTspt is not None:
        opt['datasets']['data_type']['GTspt'] = args.GTspt

    if args.spt is not None:
        opt['datasets']['data_type']['spt'] = args.spt

    if args.input_size is not None:
        opt['model']['input_size'] = args.input_size

    if args.model_type is not None:
        opt['model']['name'] = args.model_type

    if opt['phase']['train'] is not None:
        if args.train_path is not None:
            opt['datasets']['train']['args']['data_root'] = args.train_path
        if not os.path.isabs(opt['datasets']['train']['args']['data_root']):
            opt['datasets']['train']['args']['data_root'] = os.path.abspath(opt['datasets']['train']['args']['data_root'])

        if args.train_size is not None:
            opt['datasets']['train']['args']['data_len'] = args.train_size

        if args.validation_split is not None:
            opt['datasets']['train']['dataloader']['validation_split'] = args.validation_split

        if args.num_workers is not None:
            opt['datasets']['train']['dataloader']['args']['num_workers'] = args.num_workers
            opt['datasets']['train']['dataloader']['val_args']['num_workers'] = args.num_workers

        if args.epochs is not None:
            opt['model']['hyperparameters']['epochs'] = args.epochs

        if args.lr is not None:
            opt['model']['hyperparameters']['lr'] = args.lr

        if args.bs is not None:
            opt['datasets']['train']['dataloader']['args']['batch_size'] = args.bs
            opt['datasets']['train']['dataloader']['val_args']['batch_size'] = args.bs

        if args.weight_decay is not None:
            opt['model']['hyperparameters']['weight_decay'] = args.weight_decay

        if args.loss_fn is not None:
            opt['model']['hyperparameters']['loss_fn'] = args.loss_fn

        if args.lr_scheduler is not None:
            opt['model']['hyperparameters']['lr_scheduler'] = args.lr_scheduler

        if args.scheduler_step_size is not None:
            opt['model']['hyperparameters']['scheduler_args']['step_size'] = args.scheduler_step_size

        if args.scheduler_gamma is not None:
            opt['model']['hyperparameters']['scheduler_args']['scheduler_gamma'] = args.scheduler_gamma

        if args.initializer is not None:
            opt['model']['hyperparameters']['initializer'] = args.initializer

    if opt['phase']['test_sim'] is not None:
        if args.test_path is not None:
            opt['datasets']['test_sim']['args']['data_root'] = args.test_path
        if not os.path.isabs(opt['datasets']['test_sim']['args']['data_root']):
            opt['datasets']['test_sim']['args']['data_root'] = os.path.abspath(opt['datasets']['test_sim']['args']['data_root'])
        opt["datasets"]["test_sim"]["args"]["name"] = \
            os.path.splitext(os.path.basename(opt['datasets']['test_sim']['args']['data_root']))[0]
        if args.test_size is not None:
            opt['datasets']['test_sim']['args']['data_len'] = args.test_size

        if args.num_workers is not None:
            opt['datasets']['test_sim']['dataloader']['args']['num_workers'] = args.num_workers

        if args.bs is not None:
            opt['datasets']['test_sim']['dataloader']['args']['batch_size'] = args.bs

    if opt['phase']['test_exp'] is not None:
        if args.test_path is not None:
            opt['datasets']['test_exp']['args']['data_root'] = args.test_path
        if not os.path.isabs(opt['datasets']['test_exp']['args']['data_root']):
            opt['datasets']['test_exp']['args']['data_root'] = os.path.abspath(opt['datasets']['test_exp']['args']['data_root'])
        opt["datasets"]["test_exp"]["args"]["name"] = \
        os.path.splitext(os.path.basename(opt['datasets']['test_exp']['args']['data_root']))[0]

        if args.test_size is not None:
            opt['datasets']['test_exp']['args']['data_len'] = args.test_size

        if args.num_workers is not None:
            opt['datasets']['test_exp']['dataloader']['args']['num_workers'] = args.num_workers

        if args.bs is not None:
            opt['datasets']['test_exp']['dataloader']['args']['batch_size'] = args.bs

    ''' set log directory '''
    experiments_root = os.path.join(opt['exp_path']['base_dir'], opt['experiment_name'])
    mkdirs(experiments_root)
    mkdirs(os.path.join(experiments_root, opt['exp_path']['configs'])) # os.path.join(experiments_root)
    # print('results and model will be saved in {}'.format(experiments_root))
    ''' save json '''
    json_filename = []
    if opt['phase']['train']:
        json_filename.append('train')
    if opt['phase']['test_sim']:
        json_filename.append('test_sim')
    if opt['phase']['test_exp']:
        json_filename.append('test_exp')
    json_filename = '_'.join(json_filename)
    write_json(opt, '{}/{}/{}_config_{}.json'.format(experiments_root, opt['exp_path']['configs'], json_filename, get_timestamp()))
    return dict_to_nonedict(opt)
