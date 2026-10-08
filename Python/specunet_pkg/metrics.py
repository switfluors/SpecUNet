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
from scipy.signal import find_peaks
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


def _eval_two_gaussian(x, A1, b1, c1, A2, b2, c2):
    """Evaluate the 2-Gaussian model, skipping zero-amplitude components.

    A component rejected by the FWHM cut has A forced to 0 and its width blanked,
    which would make _two_gaussian divide by zero. Skipping it keeps the returned
    curve consistent with the coefficients actually reported in the tables.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if A1 > 0 and c1 > 0:
        out = out + A1 * np.exp(-((x - b1) / c1) ** 2)
    if A2 > 0 and c2 > 0:
        out = out + A2 * np.exp(-((x - b2) / c2) ** 2)
    return out


def estimate_pedestal(y, method="percentile", percentile=10.0, edge_frac=0.1):
    """Estimate the flat baseline under a 1-D spectrum.

    The 2-Gaussian model has no constant term, so any residual background floor
    has to be absorbed by widening the Gaussians -- which pushes them past the
    FWHM rejection cut and registers as a fit failure.
    """
    y = np.asarray(y, dtype=np.float64)
    if method == "none" or y.size == 0:
        return 0.0
    if method == "min":
        return float(np.nanmin(y))
    if method == "percentile":
        return float(np.nanpercentile(y, percentile))
    if method == "edges":
        k = max(1, int(round(edge_frac * y.size)))
        return float(np.nanmedian(np.concatenate([y[:k], y[-k:]])))
    raise ValueError(f"Unknown pedestal method: {method}. Use none/min/percentile/edges.")


def _initial_guess(x, y, fwhm_max):
    """Data-driven starting parameters for the 2-Gaussian fit.

    The previous fixed guess used b=mean(x) and c=std(x), i.e. a pair of
    Gaussians centred mid-axis with FWHM ~205 nm -- already within 20% of the
    250 nm rejection threshold. A fit that failed to move far from that guess
    was discarded, so broad or low-contrast spectra failed by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    span = float(x[-1] - x[0]) if x.size > 1 else 1.0
    amp = float(np.nanmax(y)) if y.size else 0.0

    if not np.isfinite(amp) or amp <= 0:
        c0 = fwhm_max / (4.0 * 2.355)
        mid = float(np.mean(x))
        return [1.0, mid, c0, 0.5, min(mid + c0, float(x[-1])), c0]

    # Width from the half-maximum crossings of the dominant peak.
    above = np.flatnonzero(y >= 0.5 * amp)
    fwhm1 = float(x[above[-1]] - x[above[0]]) if above.size >= 2 else 0.1 * span
    fwhm1 = float(np.clip(fwhm1, 0.02 * span, 0.8 * fwhm_max))
    c1 = fwhm1 / 2.355

    peaks, props = find_peaks(y, height=0.15 * amp,
                              distance=max(1, int(round(0.02 * y.size))))
    if peaks.size:
        peaks = peaks[np.argsort(props["peak_heights"])[::-1]]
    else:
        peaks = np.array([int(np.nanargmax(y))])

    b1, A1 = float(x[peaks[0]]), float(y[peaks[0]])
    if peaks.size > 1:
        b2, A2 = float(x[peaks[1]]), float(y[peaks[1]])
    else:
        # sSMLM emission spectra carry a vibronic shoulder on the red side.
        b2, A2 = b1 + fwhm1, A1 / 3.0

    b1 = float(np.clip(b1, x[0], x[-1]))
    b2 = float(np.clip(b2, x[0], x[-1]))
    return [max(A1, 1e-12), b1, c1, max(A2, 1e-12), b2, c1]


def _fit_one_spectrum(x, y, fwhm_max, maxfev, pedestal_kw):
    """Fit one spectrum with the 2-Gaussian model.

    Returns a dict of coefficients, the fitted curve, and a status describing
    exactly how the fit ended, so failures can be counted rather than silently
    turning into NaN downstream.
    """
    y = np.asarray(y, dtype=np.float64)

    pedestal = estimate_pedestal(y, **pedestal_kw)
    y_corr = np.clip(y - pedestal, 0.0, None)

    dynamic_range = float(np.nanmax(y_corr) - np.nanmin(y_corr)) if y_corr.size else 0.0
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        return {
            "A1": 0.0, "b1_nm": 0.0, "fwhm1": 0.0,
            "A2": 0.0, "b2_nm": 0.0, "fwhm2": 0.0,
            "ratio": 0.0, "curve": np.zeros_like(y_corr),
            "pedestal": pedestal, "intensity": float(np.sum(y_corr)),
            "status": "flat", "converged": False,
        }

    bounds = ([0.0, float(np.min(x)), 0.0, 0.0, float(np.min(x)), 0.0],
              [np.inf, float(np.max(x)), np.inf, np.inf, float(np.max(x)), np.inf])

    try:
        popt, _ = curve_fit(_two_gaussian, x, y_corr,
                            p0=_initial_guess(x, y_corr, fwhm_max),
                            bounds=bounds, maxfev=maxfev)
        A1, b1, c1, A2, b2, c2 = (float(v) for v in popt)
        converged = True
    except Exception:
        A1 = b1 = c1 = A2 = b2 = c2 = 0.0
        converged = False

    # The model is symmetric under exchange of its two components, so curve_fit
    # is free to return them in either order. Comparing the prediction's "first
    # peak" against the ground truth's "first peak" then silently compares two
    # different physical transitions, injecting an error the size of the peak
    # separation. Order by centre wavelength; a negligible component ranks last
    # because its position is not meaningful.
    amp_floor = 0.01 * max(A1, A2)
    both_significant = A1 >= amp_floor and A2 >= amp_floor
    if (b2 < b1) if both_significant else (A2 > A1):
        A1, b1, c1, A2, b2, c2 = A2, b2, c2, A1, b1, c1

    fwhm1, fwhm2 = c1 * 2.355, c2 * 2.355
    b1_nm, b2_nm = b1 + 500.0, b2 + 500.0
    ratio = A1 / A2 if A2 != 0 else 0.0

    rejected = []
    if fwhm1 > fwhm_max:
        fwhm1, A1, b1_nm, ratio = 0.0, 0.0, 0.0, 0.0
        rejected.append("first")
    if fwhm2 > fwhm_max:
        fwhm2, A2, b2_nm, ratio = 0.0, 0.0, 0.0, 0.0
        rejected.append("second")

    if not converged:
        status = "fit_failed"
    elif len(rejected) == 2:
        status = "rejected_both"
    elif rejected:
        status = f"rejected_{rejected[0]}"
    else:
        status = "ok"

    return {
        "A1": A1, "b1_nm": b1_nm, "fwhm1": fwhm1,
        "A2": A2, "b2_nm": b2_nm, "fwhm2": fwhm2,
        "ratio": ratio,
        "curve": _eval_two_gaussian(x, A1, b1, c1, A2, b2, c2),
        "pedestal": pedestal, "intensity": float(np.sum(y_corr)),
        "status": status, "converged": converged,
    }


def fit_two_gaussian_for_peak_metrics(rawspt, vq, wavelengths,
                                      pedestal=None, fwhm_max=250.0, maxfev=5000):
    """
    Python version of the MATLAB 2-Gaussian fitting loop.

    Parameters
    ----------
    rawspt : np.ndarray
        Ground-truth spectra, shape (len(wavelengths), N)
    vq : np.ndarray
        Predicted spectra, shape (len(wavelengths), N)
    wavelengths : np.ndarray
        1D wavelength axis in nm, shape (len(wavelengths),)
    pedestal : dict, optional
        Baseline-subtraction settings passed to `estimate_pedestal`, plus an
        `apply_to` key of "both" (default), "pred", or "none". Applying the same
        correction to both arms keeps the peak/FWHM comparison unbiased.
    fwhm_max : float
        Components wider than this (nm) are rejected, matching the MATLAB logic.
    maxfev : int
        Function-evaluation budget for `curve_fit`.

    Returns
    -------
    coefTablerawsptf : pd.DataFrame
    coefTableSpef    : pd.DataFrame
    rawsptf          : np.ndarray  # fitted GT spectra, same shape as rawspt
    spef             : np.ndarray  # fitted predicted spectra, same shape as vq
    wavelengths      : np.ndarray  # just passed through
    fit_status       : pd.DataFrame  # per-spectrum outcome for both arms
    """
    numSpectra = rawspt.shape[1]
    assert vq.shape == rawspt.shape, "vq and rawspt must have same shape"

    pedestal = dict(pedestal or {})
    apply_to = pedestal.pop("apply_to", "both")
    if apply_to not in ("both", "pred", "none"):
        raise ValueError(f"pedestal.apply_to must be both/pred/none, got {apply_to!r}")
    pedestal.setdefault("method", "percentile")
    pedestal.setdefault("percentile", 10.0)

    pred_kw = pedestal if apply_to in ("both", "pred") else {"method": "none"}
    gt_kw = pedestal if apply_to == "both" else {"method": "none"}

    # MATLAB fitted on x = wavelengths - 500 and reported b + 500; mirrored here.
    x = wavelengths - 500.0

    spef = np.zeros_like(vq, dtype=np.float64)
    rawsptf = np.zeros_like(rawspt, dtype=np.float64)

    coefTableSpef_rows = []
    coefTablerawsptf_rows = []
    status_rows = []

    centroid_spef = np.full(numSpectra, np.nan)
    centroid_rawsptf = np.full(numSpectra, np.nan)
    centroid_rawspt = np.full(numSpectra, np.nan)

    for n in range(numSpectra):
        # ---------- predicted spectrum (-> spef) ----------
        pred = _fit_one_spectrum(x, vq[:, n], fwhm_max, maxfev, pred_kw)
        spef[:, n] = pred["curve"]

        I_spef = float(np.sum(spef[:, n]))
        centroid_spef[n] = (np.sum(wavelengths * spef[:, n]) / I_spef) if I_spef > 0 else np.nan

        coefTableSpef_rows.append({
            'Spef_firstPeakValues': pred["A1"],
            'Spef_firstPeakWavelengths': pred["b1_nm"],
            'Spef_firstPeakFWHM': pred["fwhm1"],
            'Spef_secondPeakValues': pred["A2"],
            'Spef_secondPeakWavelengths': pred["b2_nm"],
            'Spef_secondPeakFWHM': pred["fwhm2"],
            'Spef_peakRatio': pred["ratio"],
            'Intensity': pred["intensity"],
            'Spef_pedestal': pred["pedestal"],
            'centroid_spef': centroid_spef[n],
        })

        # ---------- ground-truth spectrum (-> rawsptf) ----------
        gt = _fit_one_spectrum(x, rawspt[:, n], fwhm_max, maxfev, gt_kw)
        rawsptf[:, n] = gt["curve"]

        I_rawsptf = float(np.sum(rawsptf[:, n]))
        centroid_rawsptf[n] = (np.sum(wavelengths * rawsptf[:, n]) / I_rawsptf) if I_rawsptf > 0 else np.nan

        gt_corr = np.clip(rawspt[:, n] - gt["pedestal"], 0.0, None)
        I_rawspt = float(np.sum(gt_corr))
        centroid_rawspt[n] = (np.sum(wavelengths * gt_corr) / I_rawspt) if I_rawspt > 0 else np.nan

        coefTablerawsptf_rows.append({
            'rawsptf_firstPeakValues': gt["A1"],
            'rawsptf_firstPeakWavelengths': gt["b1_nm"],
            'rawsptf_firstPeakFWHM': gt["fwhm1"],
            'rawsptf_secondPeakValues': gt["A2"],
            'rawsptf_secondPeakWavelengths': gt["b2_nm"],
            'rawsptf_secondPeakFWHM': gt["fwhm2"],
            'rawsptf_peakRatio': gt["ratio"],
            'Intensity': gt["intensity"],
            'rawsptf_pedestal': gt["pedestal"],
            'centroid_rawsptf': centroid_rawsptf[n],
            'centroid_rawspt': centroid_rawspt[n],
        })

        status_rows.append({
            'Index': n,
            'pred_status': pred["status"],
            'pred_converged': pred["converged"],
            'pred_first_valid': pred["A1"] != 0,
            'pred_second_valid': pred["A2"] != 0,
            'pred_centroid_valid': np.isfinite(centroid_spef[n]),
            'pred_pedestal': pred["pedestal"],
            'gt_status': gt["status"],
            'gt_converged': gt["converged"],
            'gt_first_valid': gt["A1"] != 0,
            'gt_second_valid': gt["A2"] != 0,
            'gt_centroid_valid': np.isfinite(centroid_rawsptf[n]),
            'gt_pedestal': gt["pedestal"],
        })

    coefTableSpef = pd.DataFrame(coefTableSpef_rows)
    coefTablerawsptf = pd.DataFrame(coefTablerawsptf_rows)
    fit_status = pd.DataFrame(status_rows)

    return coefTablerawsptf, coefTableSpef, rawsptf, spef, wavelengths, fit_status

def compute_and_save_peak_metrics(
        coefTablerawsptf: pd.DataFrame,
        coefTableSpef: pd.DataFrame,
        rawsptf: np.ndarray,
        spef: np.ndarray,
        wavelengths: np.ndarray,
        output_dir: str,
        prefix: str = "Predicted_SC",
        fit_status: pd.DataFrame = None
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
        if np.all(np.isnan(v)):
            return np.nan, np.nan
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

    # ---- how many spectra each aggregate is actually built from ----
    # Every statistic above is a nanmean, so a spectrum whose fit collapsed just
    # disappears from the average instead of registering as an error. Reporting
    # the surviving count next to each figure keeps that visible.
    summary["n_spectra"] = int(N)
    for key, values in errs.items():
        if key.endswith("_sq"):
            summary[f"{key[:-3]}_n_valid"] = int(np.count_nonzero(~np.isnan(values)))
    summary["centroid_spef_n_valid"] = int(np.count_nonzero(~np.isnan(centroid_spef)))
    summary["centroid_rawspt_n_valid"] = int(np.count_nonzero(~np.isnan(centroid_rawspt)))

    if fit_status is not None and len(fit_status):
        n = len(fit_status)
        summary["fit_success_rate"] = float(fit_status["pred_centroid_valid"].mean())
        summary["fit_failure_rate"] = 1.0 - summary["fit_success_rate"]
        summary["fit_success_rate_gt"] = float(fit_status["gt_centroid_valid"].mean())
        summary["fit_converged_rate"] = float(fit_status["pred_converged"].mean())
        summary["fit_first_peak_valid_rate"] = float(fit_status["pred_first_valid"].mean())
        summary["fit_second_peak_valid_rate"] = float(fit_status["pred_second_valid"].mean())
        summary["mean_pedestal_pred"] = float(fit_status["pred_pedestal"].mean())
        summary["mean_pedestal_gt"] = float(fit_status["gt_pedestal"].mean())
        for status, count in fit_status["pred_status"].value_counts().items():
            summary[f"fit_status_{status}"] = int(count)

        print(f"[Peak Metrics] Fit success rate (prediction): "
              f"{summary['fit_success_rate'] * 100:.1f}% ({int(summary['fit_success_rate'] * n)}/{n})")
        print(f"[Peak Metrics] Fit success rate (ground truth): "
              f"{summary['fit_success_rate_gt'] * 100:.1f}%")
        print(f"[Peak Metrics] Prediction fit outcomes: "
              f"{dict(fit_status['pred_status'].value_counts())}")

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
    if fit_status is not None:
        fit_status.to_csv(os.path.join(output_dir, f"{prefix}_fit_status.csv"), index=False)

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


def compute_background_metrics(pred_bg, gt_bg, pred_img, gt_img, bg_threshold=0.01):
    """Quantify how much background survives into the denoised image.

    Two distinct defects are measured, because they behave differently:

    * ``bg_bias`` -- mean(pred_bg - gt_bg). A global offset here shifts every
      denoised image by the same amount and is correctable post-hoc.
    * ``leftover_in_bg`` -- mean intensity of pred_img over pixels the ground
      truth calls empty. This is what collapses the peak-to-background contrast
      once the 16 spatial rows are summed into a 1-D spectrum, which in turn is
      what starves the 2-Gaussian fit.

    A model can score well on the first and badly on the second.

    Parameters
    ----------
    pred_bg, gt_bg, pred_img, gt_img : np.ndarray
        Arrays shaped (N, H, W).
    bg_threshold : float
        Pixels below this fraction of each ground-truth image's peak count as
        true background.

    Returns
    -------
    dict of scalar summary statistics.
    """
    pred_bg = np.asarray(pred_bg, dtype=np.float64)
    gt_bg = np.asarray(gt_bg, dtype=np.float64)
    pred_img = np.asarray(pred_img, dtype=np.float64)
    gt_img = np.asarray(gt_img, dtype=np.float64)

    n = pred_bg.shape[0]
    axes = tuple(range(1, pred_bg.ndim))

    bias_per_image = np.mean(pred_bg - gt_bg, axis=axes)

    leftover = np.full(n, np.nan)
    contrast_pred = np.full(n, np.nan)
    contrast_gt = np.full(n, np.nan)

    for i in range(n):
        gt_peak = float(np.max(gt_img[i]))
        if not np.isfinite(gt_peak) or gt_peak <= 0:
            continue

        mask = gt_img[i] <= bg_threshold * gt_peak
        if not mask.any():
            continue

        pred_peak = float(np.max(pred_img[i]))
        leftover[i] = float(np.mean(pred_img[i][mask]))

        if pred_peak > 0:
            contrast_pred[i] = 1.0 - leftover[i] / pred_peak
        contrast_gt[i] = 1.0 - float(np.mean(gt_img[i][mask])) / gt_peak

    return {
        "bg_bias": float(np.mean(bias_per_image)),
        "bg_bias_std": float(np.std(bias_per_image)),
        "bg_abs_bias": float(np.mean(np.abs(bias_per_image))),
        "leftover_in_bg": float(np.nanmean(leftover)),
        "contrast_pred": float(np.nanmean(contrast_pred)),
        "contrast_gt": float(np.nanmean(contrast_gt)),
        "contrast_deficit": float(np.nanmean(contrast_gt) - np.nanmean(contrast_pred)),
        "bg_n_images": int(np.count_nonzero(~np.isnan(leftover))),
        # Per-image distributions, kept so downstream code can run paired tests
        # between models rather than comparing bare aggregates.
        "per_image": {
            "bias": bias_per_image,
            "leftover": leftover,
            "contrast_deficit": contrast_gt - contrast_pred,
        },
    }


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

