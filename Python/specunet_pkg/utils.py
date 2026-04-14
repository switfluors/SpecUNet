import os
import sys
import warnings
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime

from .models import UNet
from .logger import get_logger, log_print

def ignore_warnings():
    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model_path_filename(model_name, model_type='base'):
    """Generates model filename based on model_name and type."""

    if model_name == 'unet':
        model_name = "conv_UNet"

    if model_type == "traced":
        model_ext = ".pt"
    elif model_type == "onnx":
        model_ext = ".onnx"
    else:
        model_ext = ".pth"

    if model_type != '':
        model_type += '_'

    model_filename = f'{model_type}{model_name}_model{model_ext}'

    return model_filename

def load_model(models, model_folder, device, model_ext="base", base_path=None):
    """
    Loads either ONNX or .pth model based on args.model_ext.
    
    Args:
        base_path (str, optional): The absolute path to the directory containing 
                                   the model folders. If None, uses relative path.
    """
    filename = get_model_path_filename(model_folder, model_ext)
    
    if base_path:
        # Construct absolute path: base_path/model_folder/filename
        model_path = os.path.join(base_path, model_folder, filename)
    else:
        # Fallback to relative path
        model_path = os.path.join(model_folder, filename)

    # Debug print to help verify path resolution
    # print(f"DEBUG: Loading model from: {os.path.abspath(model_path)}")

    models[model_folder].load_state_dict(torch.load(model_path, map_location=device))

    return models

def total_variation_loss(x):
    """
    Computes the total variation loss for a batch of images.
    Encourages smoothness by penalizing large differences between neighboring pixels.
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def get_models(device, opt):
    """Initialize the chosen model(s)."""
    models = {}

    model_name = opt["name"]
    if model_name not in ["unet", "all"]:
        raise ValueError(f"Invalid model type: {model_name}")

    INITIALIZERS = {
        "none": None,
        "kaiming_normal": lambda m: initialize_weights(m, "kaiming_normal"),
        "kaiming_uniform": lambda m: initialize_weights(m, "kaiming_uniform"),
        "xavier_normal": lambda m: initialize_weights(m, "xavier_normal"),
        "xavier_uniform": lambda m: initialize_weights(m, "xavier_uniform"),
        "orthogonal": lambda m: initialize_weights(m, "orthogonal"),
        "normal": lambda m: initialize_weights(m, "normal"),
        "constant": lambda m: initialize_weights(m, "constant"),
    }

    # Get initializer type from config
    init_name = opt["hyperparameters"].get("initializer", "none").lower()
    initializer = INITIALIZERS.get(init_name)
    # print(f"Using initializer: {init_name}")

    if initializer is None and init_name != "none":
        raise ValueError(f"Unknown initializer: {init_name}")

    if model_name in ["unet", "all"]:
        model = UNet(opt["input_size"][0], 4, 4)
        if initializer:
            model = model.apply(initializer)
        models["unet"] = model.double().to(device)

    # print("Available models:", ", ".join(models.keys()))
    return models

def get_loss_fn(opt_model):
    if opt_model["loss_fn"] == "rmse":
        def rmse_loss(output, target):
            return torch.sqrt(torch.mean((output - target) ** 2))
        criterion = rmse_loss
    elif opt_model["loss_fn"] == "mse":
        criterion = torch.nn.MSELoss(reduction='mean')
    elif opt_model["loss_fn"] == "mae":
        criterion = torch.nn.L1Loss()
    elif opt_model["loss_fn"] == "huber":
        criterion = torch.nn.SmoothL1Loss()
    elif opt_model["loss_fn"] == "combined":
        mse = torch.nn.MSELoss()
        opt_model_loss_fn = opt_model["loss_fn_args"]
        tv_weight = opt_model_loss_fn["tv_weight"]
        def combined_loss(pred, target):
            tv = total_variation_loss(pred)
            return mse(pred, target) + tv_weight * tv
        criterion = combined_loss
    else:
        raise ValueError(f"Invalid loss function: {opt_model['loss_fn']}")

    return criterion

def get_optimizer(opt_model_hyp, model):
    if opt_model_hyp["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_model_hyp["lr"],
                                     weight_decay=opt_model_hyp["weight_decay"])
    elif opt_model_hyp["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_model_hyp["lr"],
                                      weight_decay=opt_model_hyp["weight_decay"])
    elif opt_model_hyp["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt_model_hyp["lr"],
                                    weight_decay=opt_model_hyp["weight_decay"])
    else:
        raise ValueError(f"Invalid optimizer: {opt_model_hyp['optimizer']}")

    return optimizer

def get_lrs(opt_model_hyperparameters, optimizer, train_loader):
    if opt_model_hyperparameters["lr_scheduler"] == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            step_size_up=opt_model_hyperparameters["scheduler_args"]["step_size"],
            step_size_down=opt_model_hyperparameters["scheduler_args"]["step_size"],
            gamma=opt_model_hyperparameters["scheduler_args"]["gamma"],
            base_lr=opt_model_hyperparameters["lr"],
            max_lr=opt_model_hyperparameters["lr"],
        )
    elif opt_model_hyperparameters["lr_scheduler"] == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=opt_model_hyperparameters["epochs"] * len(train_loader),
            steps_per_epoch=opt_model_hyperparameters["epochs"] * len(train_loader),
            pct_start=opt_model_hyperparameters["scheduler_args"]["pct_start"],
            max_lr=opt_model_hyperparameters["lr"]
        )
    elif opt_model_hyperparameters["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt_model_hyperparameters["scheduler_args"]["T_max"],
        )
    else:  # args.lr_scheduler == "step"
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opt_model_hyperparameters["scheduler_args"]["step_size"],
            gamma=opt_model_hyperparameters["scheduler_args"]["gamma"]
        )
    return scheduler


def initialize_weights(m, init_type):
    """Unified weight initializer for supported types."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if init_type == "kaiming_normal":
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "kaiming_uniform":
            init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "xavier_normal":
            init.xavier_normal_(m.weight)
        elif init_type == "xavier_uniform":
            init.xavier_uniform_(m.weight)
        elif init_type == "orthogonal":
            init.orthogonal_(m.weight)
        elif init_type == "normal":
            init.normal_(m.weight, mean=0.0, std=0.02)
        elif init_type == "constant":
            init.constant_(m.weight, 0.1)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder, exists_ok=True)


def start_logging(path):
    logger = get_logger(path, "Main")
    log_print(logger, "[main] Command run: python " + " ".join(sys.argv))
    log_print(logger, "[main] Start time: " + datetime.now().strftime("%m/%d/%Y %I:%M:%S %p"))
    return logger


def end_logging(logger):
    """
    Cleans up the logger, closes files, and restores standard output/error.
    """
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    print("[main] Logging successfully ended.")

def log_phase(logger, phase: str):
    bar = "=" * 50
    logger.info(f"\n{bar}\n>>> PHASE: {phase.upper()} <<<\n{bar}\n")

def get_phases(opt_phase):
    phases = ["train", "test_sim", "test_exp"]
    current_phases = []
    for phase in phases:
        if opt_phase[phase]:
            current_phases.append(phase)
    return current_phases

def get_total_time(start_time: datetime, end_time: datetime):
    total_time = end_time - start_time
    total_time_secs = total_time.total_seconds()

    hours, remainder = divmod(total_time_secs, 3600)
    mins, secs = divmod(remainder, 60)

    timestring = f"{int(hours)}h {int(mins)}m {secs:.2f}s"
    return timestring, total_time_secs


