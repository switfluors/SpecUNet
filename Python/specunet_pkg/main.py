import time
import json
import traceback
from datetime import datetime
from importlib import resources

# --- Relative Imports ---
from .utils import *
from .dataset import get_train_datasets, get_test_datasets
from .train import train
from .test import test
from .logger import log_print
from . import praser


def get_spt_array(ds_or_obj):
    """Helper to extract reference spectrum array safely from dataset or wrapper."""
    if ds_or_obj is None:
        return None
    if hasattr(ds_or_obj, "get_spt") and callable(getattr(ds_or_obj, "get_spt")):
        try:
            return ds_or_obj.get_spt()
        except Exception:
            pass
    for attr in ["spt", "GTspt", "gt_spt", "rawspt"]:
        val = getattr(ds_or_obj, attr, None)
        if val is not None:
            return val
    return None


def main(manual_args=None):
    """
    Main entry point.
    """
    project_root = os.getcwd()
    ignore_warnings()

    _summary = {
        "success": False,
        "error": None,
        "phase": None,
        "output_folder": None,
        "run_time_s": 0.0
    }

    start_time = datetime.now()
    print("[main] Start time: " + start_time.strftime("%m/%d/%Y %I:%M:%S %p"))

    opt = None
    logger = None
    summaries = {}
    output_metrics_dict = {}

    original_argv = sys.argv
    if manual_args is not None:
        sys.argv = ["specunet"] + manual_args

    try:
        args = praser.parse_args()

        if not args.config or not os.path.exists(args.config):
            print("[main] No valid config path provided. Loading default config...")
            try:
                default_config_path = resources.files('specunet_pkg').joinpath('config/SpecUNet.json')
                args.config = str(default_config_path)
                print(f"[main] Loaded default config: {args.config}")
            except Exception as e:
                print(f"[main] Warning: Could not load default config from package: {e}")

        opt = praser.parse_json(args)

        _summary["phase"] = get_phases(opt["phase"])
        _summary["output_folder"] = os.path.join(os.path.abspath(opt["exp_path"]["base_dir"]), opt["experiment_name"])

        device = torch.device(opt["device_args"]["device"])
        set_seed(opt["datasets"]["data_type"]["seed"])

        create_folder(opt["exp_path"]["base_dir"])
        print("[main] Test models saved at: ", os.path.abspath(opt["exp_path"]["base_dir"]))

        if os.path.exists(opt["exp_path"]["base_dir"]):
            os.chdir(opt["exp_path"]["base_dir"])

        opt_dataset = opt["datasets"]
        opt_model = opt["model"]
        opt_model_hyperparameters = opt_model["hyperparameters"]

        # --- Training Phase ---
        if opt["phase"]["train"]:
            logger = start_logging()
            log_phase(logger, "train")
            log_print(logger, f"[main] Using experiment name '{opt['experiment_name']}' for training...")

            log_print(logger, "[main] Loading training and validation datasets...")
            training_times = {}
            train_loader, val_loader = get_train_datasets(opt_dataset, opt_model["input_size"])
            log_print(logger, "[main] Loaded training and validation datasets!")

            create_folder(opt["experiment_name"])
            os.chdir(opt["experiment_name"])

            models = get_models(device, opt_model)
            log_print(logger, "[main] Available models:", ", ".join(models.keys()))

            for model_name, model in models.items():
                log_print(logger, f"[main] Training {model_name}...")

                # --- RESUME / FINE-TUNING WEIGHT LOADING ---
                if opt.get("resume"):
                    log_print(logger, f"[main] Resuming weights for {model_name} from: {opt['resume']}")
                    load_resume_weights(model, model_name, opt["resume"], device, logger)

                os.makedirs(model_name, exist_ok=True)
                os.chdir(model_name)

                criterion = get_loss_fn(opt_model)
                optimizer = get_optimizer(opt_model_hyperparameters, model)
                scheduler = get_lrs(opt_model_hyperparameters, optimizer, train_loader)

                start_time_train = time.time()
                train(model_name, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, device, opt)
                end_time_train = time.time()

                training_times[model_name] = end_time_train - start_time_train
                os.chdir("..")

            log_print(logger, f"[main] Training times (in seconds): {training_times}")
            log_print(logger, f"[main] Trained model saved at: {os.getcwd()}")
            os.chdir("..")

        # --- Testing Phase ---
        if opt["phase"]["test_sim"] or opt["phase"]["test_exp"]:
            test_phase = "test_sim" if opt["phase"]["test_sim"] else "test_exp"

            logger = start_logging()
            log_phase(logger, test_phase)

            experiment_abs_path = os.path.join(project_root, opt["exp_path"]["base_dir"], opt["experiment_name"])

            if os.path.exists(experiment_abs_path):
                os.chdir(experiment_abs_path)

            log_print(logger, f"[main] Using experiment name {opt['experiment_name']} for {test_phase}...")

            models = get_models(device, opt_model)
            log_print(logger, "[main] Available model(s):", ", ".join(models.keys()))

            log_print(logger, "[main] Loading test dataset...")
            test_dataset = get_test_datasets(opt_dataset, input_shape=np.array(opt_model["input_size"]), opt_phase=opt["phase"])
            log_print(logger, "[main] Loaded test dataset!")

            try:
                for model_folder in models.keys():
                    models = load_model(models, model_folder, device, base_path=experiment_abs_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model weights not found for '{model_folder}': {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights for '{model_folder}': {e}")

            # Prepare evaluation payload
            opt["is_sim_phase"] = opt["phase"]["test_sim"]
            opt["is_spectral"] = (opt["datasets"]["data_type"]["type"] == "spectral")

            ds = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
            opt["test_data"] = {
                "sptimg4_test": getattr(ds, "sptimg4_test", getattr(ds, "inputs", getattr(ds, "data", None))),
                "tbg4_test": getattr(ds, "tbg4_test", getattr(ds, "tbg", getattr(ds, "targets", None))),
                "gt_spt_test": getattr(ds, "gt_spt_test", getattr(ds, "gt_spt", getattr(ds, "GTspt", None))),
                "spt": get_spt_array(ds),
                "wavelengths": getattr(ds, "wavelengths", np.linspace(500, 800, 301)),
            }

            opt["models"] = models
            opt["test_dataset"] = test_dataset
            opt["device"] = device

            # Invoke test evaluation module
            summaries = test(
                models=models,
                test_dataset=test_dataset,
                device=device,
                logger=logger,
                opt=opt,
            )

            if opt["phase"]["test_sim"]:
                output_metrics_keys = opt_model["metrics"].get("summary_output", [])
                for summary_model_name, model_summary in (summaries or {}).items():
                    output_metrics_dict = {
                        f"{summary_model_name}.{k}": v
                        for k, v in model_summary.items()
                        if not output_metrics_keys or k in output_metrics_keys
                    }
                    _summary.update(output_metrics_dict)

        _summary["success"] = True

    except Exception as e:
        _summary["success"] = False
        _summary["error"] = str(e)
        print(f"\n[main] ERROR CAUGHT:\n{traceback.format_exc()}")

    finally:
        if manual_args is not None:
            sys.argv = original_argv

        os.chdir(project_root)

        end_time = datetime.now()
        timestring, total_time_secs = get_total_time(start_time, end_time)
        _summary["run_time_s"] = round(total_time_secs, 2)

        if logger is not None:
            log_print(logger, "[main] End time: " + end_time.strftime("%m/%d/%Y %I:%M:%S %p"))
            log_print(logger, "[main] Total time: " + timestring)
            end_logging(logger)

        print(f"\n[SUMMARY_JSON] {json.dumps(_summary)} [/SUMMARY_JSON]\n", flush=True)
        return _summary


if __name__ == "__main__":
    main()