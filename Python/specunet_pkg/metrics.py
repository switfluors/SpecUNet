import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import sys
import os
from scipy.io import savemat
import hdf5storage        # for true -v7.3
from scipy.optimize import curve_fit
import h5py


def compute_and_save_spectral_metrics(
        sptimg4_test: np.ndarray,
        YPred: np.ndarray,
        spt: np.ndarray,
        metrics: list,
        model_name: str,
        output_path: str,
        input_shape: tuple = None,
        predict_background: bool = False
):
    """
    1) Cast inputs to float
    2) Transpose/rotate so that inputs become (N,16,128) and gt_spt becomes (N,301)
       * Logic: If input_shape ends in 16 (e.g. 1, 128, 16), swap axes to get (N, 16, 128).
    3) Mean over rows 7–10 -> 128-point spectra
    4) Interpolate to 301 points
    5) Range-normalize raw and GT curves
    6) Compute available metrics
    7) Save selected metrics + normalized curves to Excel.

    Uses the pre-loaded global `spt` as ground truth.
    """

    # 1) Cast to float and reshape GT
    sptimg = sptimg4_test.astype(np.float64)  # Initial shape depends on input
    gt_spt = spt.astype(np.float64)  # (N,301) but can vary based on input

    # 2) Rotate predictions and inputs to (N,16,128) dynamically
    # We check if the last dimension of the input_shape is 16.
    # If so, we assume the data is (N, 128, 16) and needs to be swapped to (N, 16, 128).
    need_transpose = False
    if input_shape is not None and input_shape[-1] == 16:
        need_transpose = True
        # print(f"Input shape {input_shape} detected: Transposing data to (N, 16, 128).")
    # else:
        # print(f"Input shape {input_shape} detected: Keeping data as is.")

    # Apply transformation to Prediction Dictionary
    YPred_rot = YPred.astype(np.float64)

    if need_transpose:
        YPred_rot = np.swapaxes(YPred_rot, 1, 2)

    # Apply transformation to Source Image
    sptimg_rot = sptimg
    if need_transpose:
        sptimg_rot = np.swapaxes(sptimg_rot, 1, 2)

    if need_transpose:
        gt_spt = np.swapaxes(gt_spt, 0, 1)

    # Verify final shape is (N, 16, 128) to prevent downstream errors
    N, H, W = sptimg_rot.shape
    if H != 16 or W != 128:
        print(f"Warning: Expected shape (N, 16, 128) after processing, but got ({N}, {H}, {W}). Check input_shape.")

    # interpolation & centroid axes
    orig_x = np.arange(1, W + 1)  # 1...128
    xq = np.linspace(1, W, 301)  # 301 points
    wavelengths = np.linspace(500, 800, 301)  # nm

    # MATLAB rows 7–10 -> Python rows 6–9
    row_slice = slice(6, 10)

    metric_records = []
    curve_records = []

    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.zeros_like(arr)

    if predict_background:
        spec_pred = sptimg_rot - YPred_rot  # (N,16,128)
    else:
        spec_pred = YPred_rot

    for i in range(N):
        # 3) mean spectrum
        raw_curve = spec_pred[i, row_slice, :].mean(axis=0)
        gt_curve = gt_spt[i, :]

        # 4) interpolate 128 -> 301
        raw_i = np.interp(xq, orig_x, raw_curve)
        gt_i = gt_curve

        # 5) normalize both
        raw_n = normalize(raw_i)
        gt_n = normalize(gt_i)

        # 6a) Calculate all available correlation metrics
        squared_error = (raw_n - gt_n) ** 2
        mse = np.mean(squared_error)
        rmse = np.sqrt(mse)

        can_correlate = raw_n.std() > 0 and gt_n.std() > 0
        rho = spearmanr(raw_n, gt_n).correlation if can_correlate else np.nan
        r = pearsonr(raw_n, gt_n)[0] if can_correlate else np.nan

        # 6b) Calculate all available centroid metrics
        if gt_n.sum() > 0:
            centroid_gt = (wavelengths * gt_n).sum() / gt_n.sum()
        else:
            centroid_gt = wavelengths[np.argmax(gt_i)]

        centroid_raw = (wavelengths * raw_n).sum() / raw_n.sum() if raw_n.sum() > 0 else np.nan

        if centroid_gt != 0 and not np.isnan(centroid_raw):
            pct_err = abs(centroid_raw - centroid_gt) / centroid_gt * 100
        else:
            pct_err = np.nan

        # 6c) Store all calculated metrics in a dictionary for easy access
        all_metrics_available = {
            "RMSE": rmse,
            "Spearman rho": rho,
            "Pearson r": r,
            "Centroid GT (nm)": centroid_gt,
            "Centroid Raw (nm)": centroid_raw,
            "Centroid % Error": pct_err
        }

        # 6d) Select and store requested metrics
        record = {
            "Model": model_name,
            "Index": i,
        }
        for metric_name in metrics:
            if metric_name in all_metrics_available:
                record[metric_name] = all_metrics_available[metric_name]

        metric_records.append(record)

        # 6e) record curves
        base = {"Model": model_name, "Index": i}
        gt_row = {**base, "Type": "GroundTruth"}
        raw_row = {**base, "Type": "Prediction"}
        for idx, wl in enumerate(wavelengths):
            gt_row[f"{wl:.1f}nm"] = gt_n[idx]
            raw_row[f"{wl:.1f}nm"] = raw_n[idx]
        curve_records.extend([gt_row, raw_row])

        # 7) Save to Excel
    df_metrics = pd.DataFrame(metric_records)
    df_curves = pd.DataFrame(curve_records)

    ext = os.path.splitext(output_path)[1].lower()

    # Ensure directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if ext == ".csv":
        base = os.path.splitext(output_path)[0]
        metrics_path = base + "_metrics.csv"
        curves_path = base + "_curves.csv"

        df_metrics.to_csv(metrics_path, index=False)
        df_curves.to_csv(curves_path, index=False)

        print(f"[Spectrum Metrics] Saved spectral metrics to CSV path {metrics_path}")
        print(f"[Spectrum Metrics] Saved spectral curves to CSV path {curves_path}")

    elif ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)
            df_curves.to_excel(writer, sheet_name="curves", index=False)

        print(f"[Spectrum Metrics] Saved metrics and curves with {ext} format at {output_path}")

    else:
        raise ValueError(f"Unsupported output extension: {ext}. Use .csv or .xlsx")

    return df_metrics, df_curves

    # # Ensure directory exists if needed, or just save
    # with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    #     df_metrics.to_excel(writer, sheet_name="metrics", index=False)
    #     df_curves.to_excel(writer, sheet_name="curves", index=False)
    #
    # print(f"Saved metrics and curves to {output_path}")
    # return df_metrics, df_curves



def _two_gaussian(x, A1, b1, c1, A2, b2, c2):
    return A1 * np.exp(-((x - b1) / c1) ** 2) + A2 * np.exp(-((x - b2) / c2) ** 2)


def fit_two_gaussian_for_peak_metrics(rawspt, vq, wavelengths):
    """
    Python version of your MATLAB 2-Gaussian fitting loop.

    Parameters
    ----------
    rawspt : np.ndarray
        Raw spectra, shape (len(wavelengths), N)
    vq : np.ndarray
        Smoothed / processed spectra, shape (len(wavelengths), N)
    wavelengths : np.ndarray
        1D wavelength axis in nm, shape (len(wavelengths),)

    Returns
    -------
    coefTablerawsptf : pd.DataFrame
    coefTableSpef    : pd.DataFrame
    rawsptf          : np.ndarray  # fitted raw spectra, same shape as rawspt
    spef             : np.ndarray  # fitted vq spectra, same shape as vq
    wavelengths      : np.ndarray  # just passed through
    """
    numSpectra = rawspt.shape[1]
    assert vq.shape == rawspt.shape, "vq and rawspt must have same shape"

    # In MATLAB you used x and then added +500 to b1/b2.
    # Here we mimic that: x = wavelengths - 500, then store b+500 in tables.
    x = wavelengths - 500.0

    # Storage
    spef = np.zeros_like(vq, dtype=np.float64)
    rawsptf = np.zeros_like(rawspt, dtype=np.float64)

    coefTableSpef_rows = []
    coefTablerawsptf_rows = []

    centroid_spef = np.full(numSpectra, np.nan)
    centroid_rawsptf = np.full(numSpectra, np.nan)
    centroid_rawspt = np.full(numSpectra, np.nan)

    # Intensity = sum of spectrum (like your Intensity(n))
    Intensity_vq = np.sum(vq, axis=0)
    Intensity_rawspt = np.sum(rawspt, axis=0)

    for n in range(numSpectra):
        # ---------- 2-Gaussian fit for vq (-> spef) ----------
        y_vq = vq[:, n]

        # Initial guesses
        A1_0 = np.max(y_vq)
        b1_0 = np.mean(x)
        c1_0 = np.std(x)
        A2_0 = A1_0 / 2.0
        b2_0 = b1_0 + 1.0
        c2_0 = c1_0

        p0 = [A1_0, b1_0, c1_0, A2_0, b2_0, c2_0]
        bounds_lower = [0.0, np.min(x), 0.0, 0.0, np.min(x), 0.0]
        bounds_upper = [np.inf, np.max(x), np.inf, np.inf, np.max(x), np.inf]

        try:
            popt, _ = curve_fit(
                _two_gaussian, x, y_vq, p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=600
            )
            A1, b1, c1, A2, b2, c2 = popt
        except Exception:
            A1 = b1 = c1 = A2 = b2 = c2 = 0.0

        # Convert c -> FWHM (×2.355) and b -> wavelength (add 500)
        c1_fwhm = c1 * 2.355
        c2_fwhm = c2 * 2.355
        b1_nm = b1 + 500.0
        b2_nm = b2 + 500.0

        Ratio = A1 / A2 if A2 != 0 else 0.0

        # FWHM > 250 ⇒ zero out (match your MATLAB logic)
        if c1_fwhm > 250:
            c1_fwhm = 0.0
            A1 = 0.0
            b1_nm = 0.0
            Ratio = 0.0
        if c2_fwhm > 250:
            c2_fwhm = 0.0
            A2 = 0.0
            b2_nm = 0.0
            Ratio = 0.0

        # Fitted curve
        ypred_spef = _two_gaussian(x, *popt) if (A1 != 0 or A2 != 0) else np.zeros_like(y_vq)
        spef[:, n] = ypred_spef

        # Centroid of spef
        I_spef = np.sum(spef[:, n])
        if I_spef == 0:
            centroid_spef1 = np.nan
        else:
            centroid_spef1 = np.sum(wavelengths * spef[:, n]) / I_spef
        centroid_spef[n] = centroid_spef1

        coefTableSpef_rows.append({
            'Spef_firstPeakValues': A1,
            'Spef_firstPeakWavelengths': b1_nm,
            'Spef_firstPeakFWHM': c1_fwhm,
            'Spef_secondPeakValues': A2,
            'Spef_secondPeakWavelengths': b2_nm,
            'Spef_secondPeakFWHM': c2_fwhm,
            'Spef_peakRatio': Ratio,
            'Intensity': float(Intensity_vq[n]),
            'centroid_spef': centroid_spef1,
        })

        # ---------- 2-Gaussian fit for rawspt (-> rawsptf) ----------
        y_raw = rawspt[:, n]

        A1_0 = np.max(y_raw)
        b1_0 = np.mean(x)
        c1_0 = np.std(x)
        A2_0 = A1_0 / 2.0
        b2_0 = b1_0 + 1.0
        c2_0 = c1_0

        p0 = [A1_0, b1_0, c1_0, A2_0, b2_0, c2_0]

        try:
            popt_raw, _ = curve_fit(
                _two_gaussian, x, y_raw, p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=600
            )
            A1r, b1r, c1r, A2r, b2r, c2r = popt_raw
        except Exception:
            A1r = b1r = c1r = A2r = b2r = c2r = 0.0

        c1r_fwhm = c1r * 2.355
        c2r_fwhm = c2r * 2.355
        b1r_nm = b1r + 500.0
        b2r_nm = b2r + 500.0
        Ratio_r = A1r / A2r if A2r != 0 else 0.0

        if c1r_fwhm > 250:
            c1r_fwhm = 0.0
            A1r = 0.0
            b1r_nm = 0.0
            Ratio_r = 0.0
        if c2r_fwhm > 250:
            c2r_fwhm = 0.0
            A2r = 0.0
            b2r_nm = 0.0
            Ratio_r = 0.0

        ypred_raw = _two_gaussian(x, *popt_raw) if (A1r != 0 or A2r != 0) else np.zeros_like(y_raw)
        rawsptf[:, n] = ypred_raw

        # Centroid of rawsptf
        I_rawsptf = np.sum(rawsptf[:, n])
        if I_rawsptf == 0:
            centroid_rawsptf1 = np.nan
        else:
            centroid_rawsptf1 = np.sum(wavelengths * rawsptf[:, n]) / I_rawsptf
        centroid_rawsptf[n] = centroid_rawsptf1

        # Centroid of rawspt
        I_rawspt = Intensity_rawspt[n]
        if I_rawspt == 0:
            centroid_rawspt1 = np.nan
        else:
            centroid_rawspt1 = np.sum(wavelengths * rawspt[:, n]) / I_rawspt
        centroid_rawspt[n] = centroid_rawspt1

        coefTablerawsptf_rows.append({
            'rawsptf_firstPeakValues': A1r,
            'rawsptf_firstPeakWavelengths': b1r_nm,
            'rawsptf_firstPeakFWHM': c1r_fwhm,
            'rawsptf_secondPeakValues': A2r,
            'rawsptf_secondPeakWavelengths': b2r_nm,
            'rawsptf_secondPeakFWHM': c2r_fwhm,
            'rawsptf_peakRatio': Ratio_r,
            'Intensity': float(Intensity_rawspt[n]),
            'centroid_rawsptf': centroid_rawsptf1,
            'centroid_rawspt': centroid_rawspt1,
        })

    coefTableSpef = pd.DataFrame(coefTableSpef_rows)
    coefTablerawsptf = pd.DataFrame(coefTablerawsptf_rows)

    return coefTablerawsptf, coefTableSpef, rawsptf, spef, wavelengths

def compute_and_save_peak_metrics(
        coefTablerawsptf: pd.DataFrame,
        coefTableSpef: pd.DataFrame,
        rawsptf: np.ndarray,
        spef: np.ndarray,
        wavelengths: np.ndarray,
        output_dir: str,
        prefix: str = "Predicted_SC"
):
    """
    Compare peak parameters between raw and fitted spectra, compute error metrics,
    centroids, and save everything to CSV and MATLAB (-v7.3 if hdf5storage available).

    Parameters
    ----------
    coefTablerawsptf : pd.DataFrame
        Raw spectral fit coefficients (columns: rawsptf_firstPeakWavelengths, etc.)
    coefTableSpef : pd.DataFrame
        Smoothed/fitted spectral coefficients (columns: Spef_firstPeakWavelengths, etc.)
    rawsptf, spef : np.ndarray
        2D spectra arrays shaped (len(wavelengths), N)
    wavelengths : np.ndarray
        1D wavelength axis in nm
    output_dir : str
        Folder to save all results
    prefix : str
        Prefix for saved files (default: 'Predicted_SC')

    Returns
    -------
    summary : dict
        Dictionary of computed summary statistics (MSE/RMSE, centroids, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    N = rawsptf.shape[1]
    assert spef.shape == rawsptf.shape, "spef and rawsptf must have same shape"
    assert len(wavelengths) == rawsptf.shape[0], "wavelength axis mismatch"

    # ---- helper ----
    def calc_err(a, b):
        return abs(a - b), (a - b) ** 2

    # ---- storage ----
    errs = {
        "firstPeaks_wavelengths_abs": np.full(N, np.nan),
        "firstPeaks_wavelengths_sq": np.full(N, np.nan),
        "secondPeaks_wavelengths_abs": np.full(N, np.nan),
        "secondPeaks_wavelengths_sq": np.full(N, np.nan),
        "firstPeakFWHM_abs": np.full(N, np.nan),
        "firstPeakFWHM_sq": np.full(N, np.nan),
        "secondPeakFWHM_abs": np.full(N, np.nan),
        "secondPeakFWHM_sq": np.full(N, np.nan),
        "PeakRatio_abs": np.full(N, np.nan),
        "PeakRatio_sq": np.full(N, np.nan),
    }
    centroid_rawspt = np.full(N, np.nan)
    centroid_spef = np.full(N, np.nan)

    # ---- main loop ----
    for n in range(N):
        # peaks & widths
        a1 = coefTablerawsptf.loc[n, "rawsptf_firstPeakWavelengths"]
        b1 = coefTableSpef.loc[n, "Spef_firstPeakWavelengths"]
        a2 = coefTablerawsptf.loc[n, "rawsptf_secondPeakWavelengths"]
        b2 = coefTableSpef.loc[n, "Spef_secondPeakWavelengths"]
        a3 = coefTablerawsptf.loc[n, "rawsptf_firstPeakFWHM"]
        b3 = coefTableSpef.loc[n, "Spef_firstPeakFWHM"]
        a4 = coefTablerawsptf.loc[n, "rawsptf_secondPeakFWHM"]
        b4 = coefTableSpef.loc[n, "Spef_secondPeakFWHM"]
        a5 = coefTablerawsptf.loc[n, "rawsptf_peakRatio"]
        b5 = coefTableSpef.loc[n, "Spef_peakRatio"]

        # error computations with zero guards
        if a1 != 0 and b1 != 0:
            errs["firstPeaks_wavelengths_abs"][n], errs["firstPeaks_wavelengths_sq"][n] = calc_err(a1, b1)
        if a2 != 0 and b2 != 0:
            errs["secondPeaks_wavelengths_abs"][n], errs["secondPeaks_wavelengths_sq"][n] = calc_err(a2, b2)
        if a3 != 0 and b3 != 0:
            errs["firstPeakFWHM_abs"][n], errs["firstPeakFWHM_sq"][n] = calc_err(a3, b3)
        if a4 != 0 and b4 != 0:
            errs["secondPeakFWHM_abs"][n], errs["secondPeakFWHM_sq"][n] = calc_err(a4, b4)
        if a5 != 0 and b5 != 0:
            errs["PeakRatio_abs"][n], errs["PeakRatio_sq"][n] = calc_err(a5, b5)

        # centroids
        I_raw, I_fit = rawsptf[:, n].sum(), spef[:, n].sum()
        if I_raw > 0:
            centroid_rawspt[n] = np.sum(wavelengths * rawsptf[:, n]) / I_raw
        if I_fit > 0:
            centroid_spef[n] = np.sum(wavelengths * spef[:, n]) / I_fit

    # ---- stats ----
    def mse_rmse(v):
        return np.nanmean(v), np.sqrt(np.nanmean(v))

    summary = {
        "firstPeaks_wavelengths_mse": mse_rmse(errs["firstPeaks_wavelengths_sq"])[0],
        "firstPeaks_wavelengths_rmse": mse_rmse(errs["firstPeaks_wavelengths_sq"])[1],
        "secondPeaks_wavelengths_mse": mse_rmse(errs["secondPeaks_wavelengths_sq"])[0],
        "secondPeaks_wavelengths_rmse": mse_rmse(errs["secondPeaks_wavelengths_sq"])[1],
        "firstPeakFWHM_mse": mse_rmse(errs["firstPeakFWHM_sq"])[0],
        "firstPeakFWHM_rmse": mse_rmse(errs["firstPeakFWHM_sq"])[1],
        "secondPeakFWHM_mse": mse_rmse(errs["secondPeakFWHM_sq"])[0],
        "secondPeakFWHM_rmse": mse_rmse(errs["secondPeakFWHM_sq"])[1],
        "PeakRatio_mse": mse_rmse(errs["PeakRatio_sq"])[0],
        "PeakRatio_rmse": mse_rmse(errs["PeakRatio_sq"])[1],
        "mean_centroid_rawspt": np.nanmean(centroid_rawspt),
        "std_centroid_rawspt": np.nanstd(centroid_rawspt),
        "mean_centroid_spef": np.nanmean(centroid_spef),
        "std_centroid_spef": np.nanstd(centroid_spef),
    }

    # ---- error table ----
    Errortable = pd.DataFrame({
        "firstPeaks_wavelengths_abs_errors": errs["firstPeaks_wavelengths_abs"],
        "firstPeaks_wavelengths_squared_errors": errs["firstPeaks_wavelengths_sq"],
        "secondPeaks_wavelengths_abs_errors": errs["secondPeaks_wavelengths_abs"],
        "secondPeaks_wavelengths_squared_errors": errs["secondPeaks_wavelengths_sq"],
        "firstPeakFWHM_abs_errors": errs["firstPeakFWHM_abs"],
        "firstPeakFWHM_squared_errors": errs["firstPeakFWHM_sq"],
        "secondPeakFWHM_abs_errors": errs["secondPeakFWHM_abs"],
        "secondPeakFWHM_squared_errors": errs["secondPeakFWHM_sq"],
        "PeakRatio_abs_errors": errs["PeakRatio_abs"],
        "PeakRatio_squared_errors": errs["PeakRatio_sq"],
    })

    # ---- save CSVs ----
    coefTableSpef.to_csv(os.path.join(output_dir, f"{prefix}_predict_fitted_Results.csv"), index=False)
    coefTablerawsptf.to_csv(os.path.join(output_dir, f"{prefix}_raw_fitted_Results.csv"), index=False)
    Errortable.to_csv(os.path.join(output_dir, f"{prefix}_Errors.csv"), index=False)

    # ---- save MATLAB (-v7.3 if possible) ----
    mat_path = os.path.join(output_dir, f"{prefix}.mat")
    mdict = dict(
        coefTableSpef=coefTableSpef.to_dict(orient="list"),
        coefTablerawsptf=coefTablerawsptf.to_dict(orient="list"),
        Errortable=Errortable.to_dict(orient="list"),
        rawsptf=rawsptf,
        spef=spef,
        wavelengths=wavelengths,
        centroid_rawspt=centroid_rawspt,
        centroid_spef=centroid_spef,
        summary=summary,
    )

    # hdf5storage.savemat(
    #     mat_path, mdict, format='7.3',
    #     store_python_metadata=False, oned_as='row'
    # )
    # print(f"Results saved to:\n  {output_dir}\n  {mat_path}")

    # ---- save MATLAB v7.3 file ----
    mat_path = os.path.join(output_dir, f"{prefix}.mat")

    mdict = {
        "rawsptf": rawsptf,
        "spef": spef,
        "wavelengths": wavelengths,
        "centroid_rawspt": centroid_rawspt,
        "centroid_spef": centroid_spef,
    }

    try:
        save_as_v73_mat(mat_path, mdict)
        print(f"[Peak Metrics] v7.3 MAT file saved at {mat_path}")
    except Exception as e:
        print(f"[Peak Metrics] Error saving MAT file: {e}")

    return summary, Errortable

def save_as_v73_mat(filename, data_dict):
    """
    Save a dictionary to MATLAB v7.3 format (.mat) using HDF5.

    Parameters
    ----------
    filename : str
        Output .mat file path.
    data_dict : dict
        Keys become dataset names inside the MAT file.
        Values must be numpy arrays or simple scalars.
    """
    with h5py.File(filename, "w") as f:
        for key, value in data_dict.items():
            # Convert lists to arrays
            if isinstance(value, list):
                value = np.array(value)

            # Store 1D object arrays as h5py special datasets
            if value.dtype == object:
                grp = f.create_group(key)
                for i, element in enumerate(value):
                    grp.create_dataset(str(i), data=np.array(element))
            else:
                f.create_dataset(key, data=value)


def get_image_wise_metrics(metrics):
    metrics_dict = dict.fromkeys(metrics, None)
    if 'RMSE' in metrics_dict:
        def rmse_per_sample(predictions, targets):
            """Compute RMSE for each sample in a batch individually."""
            # Ensure predictions and targets are numpy arrays
            predictions = np.asarray(predictions)
            targets = np.asarray(targets)

            # Calculate squared error, keeping the sample dimension.
            # This assumes the first dimension is the batch size (N).
            squared_error = (predictions - targets) ** 2

            # Determine axes to average over (all dimensions except the first/batch dimension)
            axes_to_average = tuple(range(1, predictions.ndim))

            # Calculate mean squared error for each sample
            if not axes_to_average:  # Handle 1D arrays
                mse_per_sample = squared_error
            else:
                mse_per_sample = np.mean(squared_error, axis=axes_to_average)

            # Calculate RMSE for each sample
            rmse_values = np.sqrt(mse_per_sample)
            return rmse_values

        metrics_dict['RMSE'] = rmse_per_sample

    if 'MSE' in metrics_dict:

        def mse_per_sample(predictions, targets):
            """Compute MSE for each sample in a batch individually."""
            # Ensure predictions and targets are tensors
            # Note: This function would expect torch tensors, not numpy arrays

            # 1. Get element-wise squared error without reduction
            loss_fn = torch.nn.MSELoss(reduction='none')
            squared_error = loss_fn(predictions, targets)

            # 2. Average over all dimensions except the first (batch) dimension
            axes_to_average = tuple(range(1, squared_error.ndim))
            if not axes_to_average:  # Handle 1D case
                return squared_error
            else:
                mse_values = torch.mean(squared_error, dim=axes_to_average)

            return mse_values.cpu().numpy()  # Return as numpy array for consistency

        metrics_dict['MSE'] = mse_per_sample

    if 'MAE' in metrics_dict:  # 'mae'
        def mae_per_sample(predictions, targets):
            """Compute MAE for each sample in a batch individually."""
            # Ensure predictions and targets are tensors
            # Note: This function would expect torch tensors, not numpy arrays

            # 1. Get element-wise squared error without reduction
            loss_fn = torch.nn.L1Loss(reduction='none')
            abs_error = loss_fn(predictions, targets)

            # 2. Average over all dimensions except the first (batch) dimension
            axes_to_average = tuple(range(1, abs_error.ndim))
            if not axes_to_average:  # Handle 1D case
                mae_values = abs_error
            else:
                mae_values = torch.mean(abs_error, dim=axes_to_average)

            return mae_values.cpu().numpy()  # Return as numpy array for consistency

        metrics_dict['MAE'] = mae_per_sample

    if 'PSNR' in metrics_dict:
        def psnr_per_sample(predictions, targets):
            predictions, targets = np.asarray(predictions), np.asarray(targets)
            batch_size = predictions.shape[0]
            psnr_values = []
            for i in range(batch_size):
                data_range = targets[i].max() - targets[i].min()
                if data_range == 0:
                    psnr_values.append(np.inf if np.allclose(predictions[i], targets[i]) else 0)
                else:
                    psnr_values.append(peak_signal_noise_ratio(targets[i], predictions[i], data_range=data_range))
            return np.array(psnr_values)
        metrics_dict['PSNR'] = psnr_per_sample

    if 'SSIM' in metrics_dict:
        def ssim_per_sample(predictions, targets):
            predictions, targets = np.asarray(predictions), np.asarray(targets)
            batch_size = predictions.shape[0]
            ssim_values = []

            for i in range(batch_size):
                data_range = targets[i].max() - targets[i].min()
                if data_range == 0:
                    ssim_values.append(1.0 if np.allclose(predictions[i], targets[i]) else 0)
                    continue

                min_dim = min(targets[i].shape)
                win_size = min(min_dim, 7)
                if win_size % 2 == 0:
                    win_size -= 1

                if win_size < 2:
                    ssim_values.append(np.nan)
                    continue

                ssim_val = structural_similarity(
                    targets[i],
                    predictions[i],
                    win_size=win_size,
                    data_range=data_range,
                    multichannel=False
                )
                ssim_values.append(ssim_val)
            return np.array(ssim_values)
        metrics_dict['SSIM'] = ssim_per_sample

    return metrics_dict

def find_peak_coordinates(image_batch):
    predicted_coords = []
    for image in image_batch:
        coords = np.unravel_index(np.argmax(image, axis=None), image.shape)
        predicted_coords.append(np.array([coords]))
    return predicted_coords

def _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius):
    """
    Calculates stats by performing a direct pairwise comparison between each GT
    and its corresponding Predicted coordinate.
    """

    # Ensure the number of GT and Pred points are the same
    if len(all_gt_coords) != len(all_pred_coords):
        raise ValueError("Ground truth and prediction lists must have the same length for pairwise comparison.")

    if len(all_gt_coords) == 0:
        print("Coordinate lists are empty.")
        return 0, 0, 0, []  # TP, FP, FN, errors

    # Prepare coordinate arrays
    gt_array = np.vstack(all_gt_coords)
    pred_array = np.vstack(all_pred_coords)

    # 1. Calculate the Euclidean distance for each pair directly
    distances = np.sqrt(np.sum((gt_array - pred_array) ** 2, axis=1))

    # 2. A match is found if the distance is within the tolerance
    is_match = distances <= tolerance_radius

    # 3. Calculate TP, FP, and FN from the boolean mask
    # True Positives: The number of pairs that were a match.
    TP = np.sum(is_match)

    # Failures are both an FP and an FN, so FP will always equal FN.
    FN = len(gt_array) - TP
    FP = FN

    # 4. The localization errors are the distances of the successful matches
    localization_errors = distances[is_match].tolist()

    return TP, FP, FN, localization_errors


def get_localization_wise_metrics(metrics):
    """
    Returns a dictionary of localization metric functions. Each returned function
    takes the entire dataset of coordinates and returns a single, aggregate metric value.
    """
    metrics_dict = dict.fromkeys(metrics, None)

    if 'Jaccard Index' in metrics_dict:
        def calculate_total_jaccard(all_gt_coords, all_pred_coords, tolerance_radius=2.0):
            TP, FP, FN, _ = _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius)
            return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        metrics_dict['Jaccard Index'] = calculate_total_jaccard

    if 'Localization Recall' in metrics_dict:
        def calculate_total_recall(all_gt_coords, all_pred_coords, tolerance_radius=2.0):
            TP, FP, FN, _ = _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius)
            return TP / (TP + FN) if (TP + FN) > 0 else 0.0

        metrics_dict['Localization Recall'] = calculate_total_recall

    if 'Localization Precision' in metrics_dict:
        def calculate_total_precision(all_gt_coords, all_pred_coords, tolerance_radius=2.0):
            TP, FP, FN, _ = _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius)
            return TP / (TP + FP) if (TP + FP) > 0 else 0.0

        metrics_dict['Localization Precision'] = calculate_total_precision

    if 'Localization F1-Score' in metrics_dict:
        def calculate_total_f1_score(all_gt_coords, all_pred_coords, tolerance_radius=2.0):
            TP, FP, FN, _ = _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_dict['Localization F1-Score'] = calculate_total_f1_score

    if 'Localization Accuracy (RMSE)' in metrics_dict:
        def calculate_volumetric_rmse(all_gt_coords, all_pred_coords, tolerance_radius=2.0):
            _, _, _, errors = _calculate_pairwise_stats(all_gt_coords, all_pred_coords, tolerance_radius)
            return np.sqrt(np.mean(np.square(errors))) if errors else 0.0

        metrics_dict['Localization Accuracy (RMSE)'] = calculate_volumetric_rmse

    return metrics_dict


