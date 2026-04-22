import time
import json
import traceback
from importlib import resources

# --- Relative Imports ---
from .utils import *
from .dataset import get_train_datasets, get_test_datasets
from .train import train
from .test import test
from .logger import log_print
from . import praser


def main(manual_args=None):
    """
    Main entry point. 
    """
    project_root = os.getcwd()
    ignore_warnings()

    # 1. Initialize default summary response
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
    summary = {}
    output_metrics_dict = {}

    # --- Argument Injection Logic ---
    original_argv = sys.argv
    if manual_args is not None:
        sys.argv = ["specunet"] + manual_args

    try:
        # --- Setup & Config Parsing ---
        args = praser.parse_args()

        # Check if the user omitted the config argument or if the path doesn't exist
        if not args.config or not os.path.exists(args.config):
            print("[main] No valid config path provided. Loading default config...")
            try:
                default_config_path = resources.files('specunet_pkg').joinpath('config/SpecUNet.json')
                args.config = str(default_config_path)
                print(f"[main] Loaded default config: {args.config}")
            except Exception as e:
                print(f"[main] Warning: Could not load default config from package: {e}")

        opt = praser.parse_json(args)

        # Update summary with known configuration details
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

            create_folder(opt["experiment_name"])
            os.chdir(opt["experiment_name"])

            logger = start_logging(os.getcwd())

            log_phase(logger, "train")
            log_print(logger, f"[main] Using experiment name {opt['experiment_name']} for training...")

            log_print(logger, f"[main] Loading training and validation datasets...")
            training_times = {}
            train_loader, val_loader = get_train_datasets(opt_dataset, opt_model["input_size"])
            log_print(logger, f"[main] Loaded training and validation datasets!")

            models = get_models(device, opt_model)
            log_print(logger, f"[main] Available models:", ", ".join(models.keys()))

            for model_name, model in models.items():
                log_print(logger, f"[main] Training {model_name}...")
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

            experiment_abs_path = os.path.join(project_root, opt["exp_path"]["base_dir"], opt["experiment_name"])

            if os.path.exists(experiment_abs_path):
                os.chdir(experiment_abs_path)

            # Restart logger for testing phase
            logger = start_logging(os.getcwd())
            log_phase(logger, test_phase)

            log_print(logger, f"[main] Using experiment name {opt['experiment_name']} for {test_phase}...")

            models = get_models(device, opt_model)
            log_print(logger, f"[main] Available model(s):", ", ".join(models.keys()))

            log_print(logger, f"[main] Loading test dataset...")
            test_dataset = get_test_datasets(opt_dataset, input_shape=np.array(opt_model["input_size"]),
                                             opt_phase=opt["phase"])
            log_print(logger, f"[main] Loaded test dataset!")

            # Replace sys.exit() with standard exceptions so the outer block can catch them
            try:
                for model_folder in models.keys():
                    models = load_model(models, model_folder, device, base_path=experiment_abs_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model weights not found for '{model_folder}': {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights for '{model_folder}': {e}")

            summary = test(models, test_dataset, device, logger, opt)

            if opt["phase"]["test_sim"]:
                output_metrics_keys = opt_model["metrics"]["summary_output"]
                output_metrics_dict = {k: v for k, v in summary.items() if k in output_metrics_keys}
                _summary.update(output_metrics_dict)

        # If we reached this point without errors, the operations succeeded
        _summary["success"] = True

    except Exception as e:
        # Catch any error, mark as failed, and capture the error message
        _summary["success"] = False
        _summary["error"] = str(e)

        # Optional: Print traceback to console so you don't lose debugging context locally
        print(f"\n[main] ERROR CAUGHT:\n{traceback.format_exc()}")

    finally:
        # --- Cleanup & Summary Construction ---

        # Restore original argv
        if manual_args is not None:
            sys.argv = original_argv

        # Safely restore working directory in case it crashed while nested deep in folders
        os.chdir(project_root)

        end_time = datetime.now()
        timestring, total_time_secs = get_total_time(start_time, end_time)
        _summary["run_time_s"] = round(total_time_secs, 2)

        if logger is not None:
            log_print(logger, "[main] End time: " + end_time.strftime("%m/%d/%Y %I:%M:%S %p"))
            log_print(logger, "[main] Total time: " + timestring)
            end_logging(logger)

        # Emit structured summary
        print(f"\n[SUMMARY_JSON] {json.dumps(_summary)} [/SUMMARY_JSON]\n", flush=True)

        # Return the parsed dictionary
        return _summary


if __name__ == "__main__":
    main()