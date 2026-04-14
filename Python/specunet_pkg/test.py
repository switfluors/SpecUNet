import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys
import time
import hdf5storage


from .dataset import create_dataloader
from .metrics import *
from .logger import log_print

def save_predictions_npz_and_mat(save_dir: str,
                                 pred_bg: np.ndarray,
                                 pred_img: np.ndarray,
                                 input_img: np.ndarray,
                                 is_sim_phase: bool,
                                 gt_bg: np.ndarray = None,
                                 gt_img: np.ndarray = None,
                                 extra: dict = None):
    """
    Save prediction arrays to:
      - predictions.npz
      - predictions_v73.mat (MATLAB -v7.3)

    Arrays are saved as float32 to keep file size reasonable.
    Expected shapes typically:
      pred_bg:   (N, H, W)
      pred_img:  (N, H, W)
      input_img: (N, H, W)
      gt_bg/gt_img same if provided
    """
    os.makedirs(save_dir, exist_ok=True)

    # Cast to float32 (safe + smaller)
    pred_bg_f   = np.asarray(pred_bg, dtype=np.float32)
    pred_img_f  = np.asarray(pred_img, dtype=np.float32)
    input_img_f = np.asarray(input_img, dtype=np.float32)

    mdict = {
        "pred_bg": pred_bg_f,
        "pred_img": pred_img_f,
        "input_img": input_img_f,
    }

    if is_sim_phase:
        if gt_bg is not None:
            mdict["gt_bg"] = np.asarray(gt_bg, dtype=np.float32)
        if gt_img is not None:
            mdict["gt_img"] = np.asarray(gt_img, dtype=np.float32)

    if extra:
        # be careful: only add simple numpy arrays / scalars / strings
        mdict.update(extra)

    # ---- NPZ ----
    npz_path = os.path.join(save_dir, "predictions.npz")
    np.savez(npz_path, **mdict)

    # ---- MAT v7.3 ----
    mat_path = os.path.join(save_dir, "predictions_v73.mat")
    hdf5storage.savemat(
        mat_path,
        mdict,
        format="7.3",
        oned_as="row"
    )

    return npz_path, mat_path

def save_rmse_figures(model_data, logger):
    """Saves two RMSE figures from model data."""

    def plot_rmse_figure(title, filename, datasets, titles, filtered=False):
        """Helper function to create and save a single RMSE plot."""
        plt.figure(figsize=(4 * (1 + len(datasets)), 8 if not filtered else 4))

        # Determine the number of subplots and which data to use
        num_subplots = len(datasets)

        for i, data in enumerate(datasets):
            plt.subplot(num_subplots, 1, i + 1)

            # Plotting logic for both filtered and unfiltered
            if filtered:
                plt.hist(data, bins=30, color='green', alpha=0.7)
                mean_val, std_val = np.mean(data), np.std(data)
                precision = ".4f"

                plt.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:{precision}}')
                plt.axvline(mean_val + std_val, color='blue', linestyle='--', linewidth=2,
                            label=f'+1 Std: {mean_val + std_val:{precision}}')
                plt.axvline(mean_val - std_val, color='blue', linestyle='--', linewidth=2,
                            label=f'-1 Std: {mean_val - std_val:{precision}}')
                plt.legend()

                log_print(logger, f"[test_sim] {titles[i]} (Filtered): Mean={mean_val:{precision}}, Std={std_val:{precision}}")
            else:
                plt.hist(data, bins=30, color='blue', alpha=0.7)

            plt.title(titles[i])
            plt.xlabel('RMSE Value')
            plt.ylabel('Counts')
            plt.grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename)
        plt.close()

    # Get RMSE data and titles for the models
    datasets = [data["rmse"] for data in model_data.values()]
    titles = [data["title"] for data in model_data.values()]

    # First RMSE figure (unfiltered)
    plot_rmse_figure(
        title='Comparison of RMSE Across Different Methods',
        filename="rmse_figure_1.png",
        datasets=datasets,
        titles=titles,
        filtered=False
    )

    # Second figure (with filtering)
    filtered_datasets = [data[data <= 15] for data in datasets]
    plot_rmse_figure(
        title='Comparison of RMSE Across Different Methods (Filtered)',
        filename="rmse_figure_2.png",
        datasets=filtered_datasets,
        titles=titles,
        filtered=True
    )

    log_print(logger, f"[test_sim] RMSE Figures Saved at: {os.path.abspath(os.getcwd())}")

def save_rep_samples(model_name, data_to_plot, is_sim_phase, opt, logger, input_shape=None):
    """
    Saves and visualizes representative samples.

    Args:
        model_name (str): Name of the model (e.g., 'unet').
        data_to_plot (dict): A dictionary containing all data arrays to save/plot.
        is_sim_phase (bool): True if running in 'test_sim' phase.
        opt (dict): The options dictionary.
        logger (object): The logger object.
        input_shape (tuple): The shape of the input data (e.g., (1, 16, 128)).
    """

    def normalize_to_float64(img):
        img = img.astype(np.float64)
        img -= img.min()
        img /= img.max() + 1e-8
        return img

    # Helper function for saving a single image as a TIFF
    def save_tiff_image(img, save_path, normalize):
        if normalize:
            img = normalize_to_float64(img)
        img_uint16 = (img * 65535).clip(0, 65535).astype(np.uint16)
        Image.fromarray(img_uint16, mode='I;16').save(save_path)

    rep_indices = list(range(5))
    is_spectra = opt['datasets']['data_type']['type'] == 'spectral'
    save_tiff_images = opt['exp_path']['save_tiff_images']

    # Configure plotting and saving based on the phase flag
    if is_sim_phase:
        title_suffix = "Simulated Data"
        num_cols = 5
        images_to_plot = ['pred_bg', 'gt_bg', 'pred_img', 'gt_img', 'input_img']
        titles_to_plot = ['Predicted Background', 'Ground Truth Background', 'Predicted Spectra',
                          'Ground Truth Spectra', 'Original Spectra']
        save_results_key = 'sim_results'
        data_name_key = 'test_sim'
    else:
        title_suffix = "Experimental Data"
        num_cols = 3
        images_to_plot = ['pred_bg', 'pred_img', 'input_img']
        titles_to_plot = ["Predicted Background", "Predicted Spectra", "Original Spectra"]
        save_results_key = 'exp_results'
        data_name_key = 'test_exp'

    fig, axes = plt.subplots(len(rep_indices), num_cols, figsize=(num_cols * 10, len(rep_indices) * (3 if is_spectra else 10)))
    fig.suptitle(
        f"Conventional UNet: Representative Samples ({title_suffix})",
        fontsize=36, fontweight='bold')

    save_dir = os.path.join(model_name, opt["exp_path"][save_results_key],
                            opt["datasets"][data_name_key]["args"]["name"])
    os.makedirs(save_dir, exist_ok=True)

    norm = opt["datasets"]["data_type"]["norm"]

    for i, idx in tqdm(enumerate(range(len(data_to_plot['YPred']))), total=len(data_to_plot['YPred']),
                       desc="Saving figures", file=sys.__stdout__):
        # Extract images for the current index
        pred_bg = data_to_plot['YPred'][idx]
        input_img = data_to_plot['sptimg4_test'][idx]

        # Ensure shape is (1, 16, 128) if input is (1, 128, 16)
        if input_shape == (1, 128, 16):
            pred_bg = np.transpose(pred_bg, (0, 2, 1))
            input_img = np.transpose(input_img, (0, 2, 1))

        pred_img = input_img - pred_bg

        # Build the map of images to save
        save_map = {
            f"Input_{idx}.tif": input_img,
            f"Out_BG_{idx}.tif": pred_bg,
            f"Out_{idx}.tif": pred_img
        }
        if is_sim_phase:
            gt_bg = data_to_plot['tbg4_test'][idx]
            gt_img = data_to_plot['gt_spt_test'][idx]

            # Ensure shape is (1, 16, 128) if input is (1, 128, 16)
            if input_shape == (1, 128, 16):
                gt_bg = np.transpose(gt_bg, (0, 2, 1))
                gt_img = np.transpose(gt_img, (0, 2, 1))

            save_map.update({f"GT_BG_{idx}.tif": gt_bg, f"GT_{idx}.tif": gt_img})

        if save_tiff_images:
            for fname, img in save_map.items():
                save_tiff_image(img, os.path.join(save_dir, fname), norm)

        if i < len(rep_indices):
            # Define images and titles for plotting based on phase
            if is_sim_phase:
                images_to_plot = [pred_bg, gt_bg, pred_img, gt_img, input_img]
                titles_to_plot = ['Predicted Background', 'Ground Truth Background', 'Predicted Spectra',
                                  'Ground Truth Spectra', 'Original Spectra']
            else:
                images_to_plot = [pred_bg, pred_img, input_img]
                titles_to_plot = ["Predicted Background", "Predicted Spectra", "Original Spectra"]

            # log_print(logger, f"Stats for Image {idx}")
            for j, img in enumerate(images_to_plot):
                # log_print(logger, f"{titles_to_plot[j]} {idx}: Min={np.min(img):.6f}, Max={np.max(img):.6f}")

                ax = axes[i, j] if len(rep_indices) > 1 else axes[j]
                vis_img = normalize_to_float64(img) if norm else img
                ax.imshow(vis_img, aspect='auto', cmap='gray')
                ax.set_title(f"{titles_to_plot[j]} {idx}", fontsize=20)
                ax.axis('off')

    # Final plot save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path_base = "sim" if is_sim_phase else "exp"
    five_rep_samples_path = os.path.join(model_name, "5_rep_samples")
    os.makedirs(five_rep_samples_path, exist_ok=True)
    fig_path = os.path.join(five_rep_samples_path,
                            f"{fig_path_base}_{opt['datasets'][data_name_key]['args']['name']}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    test_phase = "[test_sim]" if is_sim_phase else "[test_exp]"
    log_print(logger, f"{test_phase} Representative Sample Figure Saved to {fig_path}")
    if save_tiff_images:
        log_print(logger, f"{test_phase} All TIFF images saved to: {save_dir}")


def test(models, test_dataset, device, logger, opt):
    test_loader = create_dataloader(test_dataset, opt['datasets']['test_sim']['dataloader']['args'])
    is_sim_phase = opt['phase']['test_sim']
    test_phase = "test_sim" if is_sim_phase else "test_exp"
    is_spectral = opt['datasets']['data_type']['type'] == 'spectral'
    data_name_key = 'test_sim' if is_sim_phase else 'test_exp'

    summary = None

    if not is_sim_phase and not opt['phase']['test_exp']:
        log_print(logger, "No test phase specified. Exiting.")
        return

    for model in models.values():
        model.eval()

    model_outputs = {name: [] for name in models}
    model_times = {name: [] for name in models}
    sptimg4_list, tbg4_list, gt_spt_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing on data", file=sys.__stdout__):
            sptimg4_batch = batch[0].to(device)
            batch_size = sptimg4_batch.size(0)
            sptimg4_list.append(sptimg4_batch.cpu().numpy())

            if is_sim_phase:
                tbg4_list.append(batch[1].cpu().numpy())
                gt_spt_list.append(batch[2].cpu().numpy())

            for model_name in models:
                start_time = time.time()
                output = models[model_name](sptimg4_batch).cpu()
                end_time = time.time()
                total_time = end_time - start_time

                time_per_sample = total_time / batch_size
                model_times[model_name].append(time_per_sample)
                model_outputs[model_name].append(output)

    log_print(logger, f"[{test_phase}] Model Inference Times:")
    for model_name, times in model_times.items():
        avg_time_ms = np.mean(times) * 1000  # Already per sample
        log_print(logger, f"[{test_phase}] {model_name}: {avg_time_ms:.3f} ms / sample")

    # Convert all lists to NumPy arrays
    sptimg4_test = np.squeeze(np.concatenate(sptimg4_list, axis=0), axis=1)
    spt = test_dataset.get_spt()

    # Save data
    for model_name, y_pred_list in model_outputs.items():
        model_outputs[model_name] = np.squeeze(torch.cat(y_pred_list, dim=0).numpy(), axis=1)

    if is_spectral and is_sim_phase:
        log_print(logger, "[test_sim] Computing and saving spectral metrics...")
        image_spectral_wise_metrics = (opt['model']['metrics']['image_wise']
                                       + opt['model']['metrics']['spectral_wise'])

        for model_name, model_output in model_outputs.items():
            spectrum_output_dir = os.path.join(
                model_name,
                opt["exp_path"].get("spectrum_metrics", "spectrum_metrics")
            )
            os.makedirs(spectrum_output_dir, exist_ok=True)

            output_path = os.path.join(
                spectrum_output_dir,
                "spectrum_wise_metrics_no_fit.csv"
            )
            log_print(logger, "[Spectrum Metrics] Saving spectrum metrics...")
            compute_and_save_spectral_metrics(
                sptimg4_test,
                model_output,
                spt,
                model_name=model_name,
                metrics=image_spectral_wise_metrics,
                output_path=output_path,
                input_shape=opt['model']['input_size'],
                predict_background=True
            )

            log_print(logger, f"[Spectrum Metrics] saved to: {output_path}")

    if is_sim_phase:
        tbg4_test = np.squeeze(np.concatenate(tbg4_list, axis=0), axis=1)
        gt_spt_test = np.squeeze(np.concatenate(gt_spt_list, axis=0), axis=1)
    else:
        tbg4_test, gt_spt_test = None, None

    image_wise_metrics = get_image_wise_metrics(opt['model']['metrics']['image_wise'])
    localization_metrics = get_localization_wise_metrics(opt['model']['metrics']['localization_wise'])

    model_data = {}
    for model_name, y_pred_list in model_outputs.items():
        y_pred = y_pred_list

        # Prepare the dictionary for the saving functions
        data_to_plot = {
            'YPred': y_pred,
            'sptimg4_test': sptimg4_test,
        }

        if is_sim_phase:
            data_to_plot.update({
                'tbg4_test': tbg4_test,
                'gt_spt_test': gt_spt_test,
            })

        # y_pred is background prediction
        pred_bg = y_pred
        input_img = sptimg4_test
        pred_img = input_img - pred_bg

        # Optional GT if sim
        gt_bg = tbg4_test if is_sim_phase else None
        gt_img = gt_spt_test if is_sim_phase else None

        # Save folder: unet/results/<dataset_name>
        dataset_name = opt["datasets"][data_name_key]["args"]["name"]
        results_dir = os.path.join(model_name, "results", dataset_name)

        # (Optional) save some metadata too
        extra = {
            "is_sim_phase": np.array([1 if is_sim_phase else 0], dtype=np.int32),
        }
        log_print(logger, f"[{test_phase}] Saving predictions to NPZ and MAT file format...")
        npz_path, mat_path = save_predictions_npz_and_mat(
            save_dir=results_dir,
            pred_bg=pred_bg,
            pred_img=pred_img,
            input_img=input_img,
            is_sim_phase=is_sim_phase,
            gt_bg=gt_bg,
            gt_img=gt_img,
            extra=extra
        )

        log_print(logger, f"[{test_phase}] Saved predictions as NPZ file: {npz_path}")
        log_print(logger, f"[{test_phase}] Saved predictions as MAT v7.3 file: {mat_path}")

        # Call the refactored save_rep_samples with a single phase flag
        save_rep_samples(
            model_name,
            data_to_plot,
            is_sim_phase,
            opt,
            logger,
            input_shape=opt['model']['input_size']
        )

        # log_print(logger, "Representative samples saved.")


        image_wise_metrics_results = dict.fromkeys(image_wise_metrics.keys(), None)
        if is_sim_phase:
            for metric_name in image_wise_metrics.keys():
                image_wise_metrics_results[metric_name] = image_wise_metrics[metric_name](y_pred, tbg4_test)
            # Only calculate RMSE if ground truth is available
            if image_wise_metrics['RMSE']:
                # rmse = metrics_results["RMSE"](y_pred, tbg4_test)
                model_data[model_name] = {
                    "rmse": image_wise_metrics_results['RMSE'],
                    "title": f"{'Conventional U-Net' if model_name == 'unet' else 'Conv. Att. U-Net'} Background RMSE",
                }
                # log_print(logger, f"{model_data[model_name]['title'].replace(' Background RMSE', '')}")

        # log_print(logger, "Image-wise metrics computed.")
        # for key, value in image_wise_metrics_results.items():
        #     avg = np.mean(value)
        #     log_print(logger, f"{key}: average = {avg}")

        for key, value in image_wise_metrics_results.items():
            # No GT, so results will be None. Skip gracefully.
            if value is None:
                log_print(logger, f"[test_exp] {key}: average = N/A (no ground truth in test_exp)")
                continue

            # If value is a list/tuple, drop any None entries
            if isinstance(value, (list, tuple)):
                value = [v for v in value if v is not None]
                if len(value) == 0:
                    log_print(logger, f"{key}: average = N/A (empty)")
                    continue

            avg = float(np.mean(value))
            # log_print(logger, f"{key}: average = {avg}")

        if is_spectral and is_sim_phase:

            print("[Peak Metrics] Computing peak-wise gaussian fitting...")


            # --- 1. We already HAVE ground-truth spectra ---
            # spt has shape (N, 301)
            wavelengths = np.linspace(500, 800, 301)
            # rawspt = spt.astype(np.float64).T  # (301, N)
            # spt expected to be (N, 301)
            if spt.shape[1] == 301:
                rawspt = spt.astype(np.float64).T  # (301, N)
            elif spt.shape[0] == 301:
                rawspt = spt.astype(np.float64)  # already (301, N)
            else:
                raise ValueError(f"spt shape invalid: {spt.shape}")
            # log_print(logger, f"Corrected rawspt shape: {rawspt.shape}")

            # --- 2. Build predicted spectra vq using the spectral extraction logic ---
            sptimg = sptimg4_test.astype(np.float64)  # (N,128,16)
            pred_bg = y_pred.astype(np.float64)  # (N,128,16)

            sptimg_rot = np.swapaxes(sptimg, 1, 2)  # (N,16,128)
            pred_bg_rot = np.swapaxes(pred_bg, 1, 2)  # (N,16,128)
            spec_pred = sptimg_rot - pred_bg_rot  # (N,16,128)

            N_samples, _, W = spec_pred.shape
            orig_x = np.arange(1, W + 1)
            xq = np.linspace(1, W, 301)
            row_slice = slice(6, 10)

            vq = np.zeros((301, N_samples), dtype=np.float64)
            for i in range(N_samples):
                raw_curve = spec_pred[i, row_slice, :].mean(axis=0)
                vq[:, i] = np.interp(xq, orig_x, raw_curve)

            # --- 3. Align in case Dataset length differs ---
            N_gt = rawspt.shape[1]
            N_pred = vq.shape[1]
            if N_gt != N_pred:
                N_common = min(N_gt, N_pred)
                log_print(logger,
                          f"[Peak Metrics] Using first {N_common} samples (GT={N_gt}, Pred={N_pred})")
                rawspt = rawspt[:, :N_common]
                vq = vq[:, :N_common]

            # log_print(logger, f"rawspt shape: {rawspt.shape}")
            # log_print(logger, f"vq shape: {vq.shape}")

            # --- 4. Run your MATLAB-equivalent 2-Gaussian peak fitting ---
            coefTablerawsptf, coefTableSpef, rawsptf, spef, wavelengths = \
                fit_two_gaussian_for_peak_metrics(rawspt, vq, wavelengths)

            # --- 5. Save peak metrics ---
            # peak_output_dir = os.path.join(
            #     model_name,
            #     opt["exp_path"].get("peak_metrics", "peak_metrics"),
            #     opt["datasets"][data_name_key]["args"]["name"],
            # )
            peak_output_dir = os.path.join(
                # opt["exp_path"]["base_dir"],
                model_name,
                opt["exp_path"].get("peak_metrics", "peak_metrics")
            )

            print("[Peak Metrics] Computing and saving peak-wise metrics...")

            summary, Errortable = compute_and_save_peak_metrics(
                coefTablerawsptf=coefTablerawsptf,
                coefTableSpef=coefTableSpef,
                rawsptf=rawsptf,
                spef=spef,
                wavelengths=wavelengths,
                output_dir=peak_output_dir,
                prefix="Peak_wise_2_gaussian"
            )

            log_print(logger, f"[Peak Metrics] Peak metrics saved to: {peak_output_dir}")

        if not is_spectral:
            gt_coords = find_peak_coordinates(tbg4_test)
            pred_coords = find_peak_coordinates(y_pred)

            localization_metrics_results = dict.fromkeys(localization_metrics.keys(), None)

            for metric_name in localization_metrics.keys():
                localization_metrics_results[metric_name] = localization_metrics[metric_name](gt_coords, pred_coords)

            log_print(logger, "[test_exp] Localization metrics computed.")
            for key, value in localization_metrics_results.items():
                avg = np.mean(value)
                log_print(logger, f"[test_exp] {key}: average = {avg}")

    if is_sim_phase:
        log_print(logger, "[test_sim] Saving RMSE figures...")
        save_rmse_figures(model_data, logger)

    return summary