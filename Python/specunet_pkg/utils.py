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


def background_region_loss(pred_bg, inputs, gt_img, threshold=0.01):
    """Penalize signal the model leaves behind in pixels that should be empty.

    The network regresses the background, so the denoised image is
    ``inputs - pred_bg``. Plain MSE against the background target averages over
    the whole frame and barely notices a thin residual spread across the empty
    region -- but that residual is exactly what accumulates when the 16 spatial
    rows are collapsed into a 1-D spectrum, producing the pedestal that starves
    the downstream 2-Gaussian fit.

    Ground-truth-empty pixels are those below `threshold` times each image's own
    peak. Images with no ground-truth signal at all are skipped, since their
    mask would otherwise cover the entire frame.
    """
    pred_img = inputs - pred_bg
    dims = tuple(range(1, gt_img.ndim))

    peak = gt_img.amax(dim=dims, keepdim=True)
    mask = (gt_img <= threshold * peak) & (peak > 0)

    count = mask.sum()
    if count == 0:
        return pred_bg.sum() * 0.0  # keeps dtype/device and the autograd graph

    return (pred_img.square() * mask).sum() / count


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

    loss_args = opt_model.get("loss_fn_args") or {}
    bg_weight = float(loss_args.get("bg_weight") or 0.0)
    bg_threshold = float(loss_args.get("bg_mask_threshold") or 0.01)

    def criterion_with_background(output, target, inputs=None, gt_img=None):
        loss = criterion(output, target)
        if bg_weight > 0 and inputs is not None and gt_img is not None:
            loss = loss + bg_weight * background_region_loss(output, inputs, gt_img, bg_threshold)
        return loss

    if bg_weight > 0:
        print(f"[utils] Background-region loss enabled "
              f"(weight={bg_weight}, mask threshold={bg_threshold} x per-image peak)")

    return criterion_with_background


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


def load_resume_weights(model, model_name, resume_path, device, logger=None):
    """
    Loads pre-trained weights into the model for fine-tuning.
    Handles folder directories (for 'unet', 'all')
    as well as direct .pth file paths safely.
    """
    expected_filename = get_model_path_filename(model_name, "base")

    if os.path.isfile(resume_path):
        file_basename = os.path.basename(resume_path)
        if model_name not in file_basename.lower() and get_model_path_filename(model_name, "base") != file_basename:
            raise ValueError(
                f"[resume] Cannot load file '{file_basename}' into model '{model_name}'. "
                f"Architecture mismatch. Please pass the experiment folder path instead when using --model_type all."
            )
        target_path = resume_path

    elif os.path.isdir(resume_path):
        # 1. Standard hierarchy: <resume_path>/<model_name>/<model_filename>
        target_path = os.path.join(resume_path, model_name, expected_filename)

        # 2. Fallback direct hierarchy: <resume_path>/<model_filename>
        if not os.path.exists(target_path):
            target_path = os.path.join(resume_path, expected_filename)
    else:
        raise FileNotFoundError(f"[resume] Resume path '{resume_path}' does not exist.")

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"[resume] Could not find checkpoint file for '{model_name}' at: {target_path}")

    log_msg = f"[resume] Loading pre-trained weights for '{model_name}' from: {os.path.abspath(target_path)}"
    if logger:
        log_print(logger, log_msg)
    else:
        print(log_msg)

    state_dict = torch.load(target_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def create_folder(folder):
    os.makedirs(folder, exist_ok=True)


def start_logging():
    logger = get_logger("Main")
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


