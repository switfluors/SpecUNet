"""Publication-style summary figure for test_sim results.

Builds a multi-panel comparison of every evaluation metric for the models that
were tested. If two models ran, each panel shows them side by side with a paired
significance test; if only one ran, the same layout renders that model alone.

Because both models see the identical set of spectra, every comparison is
*paired*: the Wilcoxon signed-rank test on per-sample differences, plus a
matched-pairs rank-biserial effect size r. At n = 5000 almost any systematic
difference reaches significance, so r (not P) is what indicates whether a
difference is large enough to matter. Binary outcomes (fit success) use
McNemar's exact test instead, which is the paired test for proportions.

Styling follows Nature figure guidelines: sans-serif type at 5-7 pt, thin axes
with top/right spines removed, a colourblind-safe palette, bold lower-case
panel letters, and a 183 mm double-column canvas saved as both PNG and PDF.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata, binomtest

DISPLAY_NAMES = {
    "unet": "UNet",
}

# Wong colourblind-safe palette
MODEL_COLORS = {
    "unet": "#0072B2",
}
FALLBACK_COLORS = ["#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # keep text editable in vector output
    "ps.fonttype": 42,
}

STAT_FS = 4.4       # font size for in-panel statistics
LOWER_IS_BETTER = object()


def _model_color(name, index):
    return MODEL_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _finite(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _fmt_v(v):
    """Compact fixed-width-ish value formatting across very different scales."""
    if v is None or not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a == 0:
        return "0"
    # Peak-ratio errors can reach ~1e16 when a fitted amplitude approaches zero,
    # so large magnitudes must not be printed in full or they wreck the layout.
    if a >= 1e5:
        return f"{v:.1e}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.001:
        return f"{v:.4f}".rstrip("0")
    return f"{v:.1e}"


def _robust_ylim(data_list):
    """Axis limits from whisker extent, ignoring the mean marker.

    Several of these metrics are so heavily right-skewed that the mean lies far
    outside the box (peak-ratio error is the extreme case). Letting matplotlib
    autoscale to the mean marker would compress every box to a flat line, so the
    axis is driven by the whiskers and an off-scale mean is simply clipped --
    its numeric value is still printed under the box.
    """
    los, his = [], []
    for d in data_list:
        d = np.asarray(d, dtype=np.float64)
        if d.size == 0:
            continue
        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3 - q1
        los.append(max(float(np.min(d)), q1 - 1.5 * iqr))
        his.append(min(float(np.max(d)), q3 + 1.5 * iqr))
    if not los:
        return None
    lo, hi = min(los), max(his)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "P n/a"
    if p < 1e-99:
        return "P < 1e-99"
    if p < 0.001:
        return f"P = {p:.0e}"
    if p < 0.01:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def _paired_stats(a, b):
    """Wilcoxon signed-rank test + matched-pairs rank-biserial effect size.

    `a` and `b` are per-sample values for the two models, aligned by index.
    Returns None when the pairing is too small or the two arms are identical.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return None
    a, b = a[:n], b[:n]
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 10:
        return None

    d = b - a
    nz = d[d != 0]
    if nz.size == 0:
        return None

    try:
        _, p = wilcoxon(a, b)
    except Exception:
        return None

    # rank-biserial r in [-1, 1]: +ve means b > a
    signed = np.sum(np.sign(nz) * rankdata(np.abs(nz)))
    r = signed / (nz.size * (nz.size + 1) / 2.0)
    return {"p": float(p), "r": float(r), "n": int(a.size)}


def _mcnemar(a_bool, b_bool):
    """Exact McNemar test for paired binary outcomes."""
    a = np.asarray(a_bool).astype(bool).ravel()
    b = np.asarray(b_bool).astype(bool).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return None
    a, b = a[:n], b[:n]
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return {"p": 1.0, "n01": 0, "n10": 0, "n": n}
    try:
        p = binomtest(n01, n01 + n10, 0.5).pvalue
    except Exception:
        return None
    return {"p": float(p), "n01": n01, "n10": n10, "n": n}


def _panel_letter(ax, letter):
    ax.text(-0.30, 1.14, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left")


def _empty(ax, message="n/a"):
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", color="0.5")
    ax.set_xticks([])
    ax.set_yticks([])


def _sig_bracket(ax, x1, x2, label, level=0):
    """Draw a significance bracket, making headroom above the data."""
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if span <= 0:
        return
    y = y1 + span * (0.06 + 0.16 * level)
    tick = span * 0.03
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
            lw=0.5, c="black", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + span * 0.01, label,
            ha="center", va="bottom", fontsize=STAT_FS, clip_on=False)
    ax.set_ylim(y0, y + span * 0.13)


def _draw_boxes(ax, data, positions, colors):
    """Boxplot with per-box colours; whiskers at 1.5 IQR, outliers hidden.

    The mean is drawn as a white diamond so both central-tendency measures are
    visible: these distributions are right-skewed, so mean and median diverge.
    """
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, showmeans=True,
                    medianprops=dict(color="black", lw=0.9),
                    meanprops=dict(marker="D", markersize=2.2,
                                   markerfacecolor="white",
                                   markeredgecolor="black",
                                   markeredgewidth=0.4),
                    boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                    capprops=dict(lw=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    return bp


def _stat_label(name, vals):
    return f"{name}\nMd {_fmt_v(np.median(vals))}\nM {_fmt_v(np.mean(vals))}"


def _box_panel(ax, per_model_values, models, ylabel, title):
    """One box per model for a single metric, with paired test between them."""
    data, colors, labels, used = [], [], [], []
    for i, m in enumerate(models):
        vals = _finite(per_model_values.get(m, []))
        if vals.size:
            data.append(vals)
            colors.append(_model_color(m, i))
            labels.append(_stat_label(DISPLAY_NAMES.get(m, m), vals))
            used.append(m)
    if not data:
        _empty(ax)
        return

    _draw_boxes(ax, data, list(range(len(data))), colors)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=3)

    lim = _robust_ylim(data)
    if lim:
        ax.set_ylim(*lim)

    if len(used) == 2:
        st = _paired_stats(per_model_values[used[0]], per_model_values[used[1]])
        if st:
            _sig_bracket(ax, 0, 1, f"{_fmt_p(st['p'])}, r = {st['r']:+.2f}")


def _grouped_box_panel(ax, per_model_groups, models, group_labels, ylabel, title):
    """len(group_labels) subgroups on the x-axis, one box per model in each."""
    n = len(models)
    step = n + 1
    data, positions, colors = [], [], []
    medians = {}
    for g, _ in enumerate(group_labels):
        for i, m in enumerate(models):
            groups = per_model_groups.get(m)
            vals = _finite(groups[g]) if groups is not None else np.array([])
            if vals.size:
                data.append(vals)
                positions.append(g * step + i)
                colors.append(_model_color(m, i))
                medians[(g, i)] = (np.median(vals), np.mean(vals))
    if not data:
        _empty(ax)
        return

    _draw_boxes(ax, data, positions, colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=3)

    lim = _robust_ylim(data)
    if lim:
        ax.set_ylim(*lim)

    tick_labels = []
    centers = []
    for g, gl in enumerate(group_labels):
        centers.append(g * step + (n - 1) / 2.0)
        parts = [gl]
        for i, m in enumerate(models):
            if (g, i) in medians:
                md, mn = medians[(g, i)]
                parts.append(f"{DISPLAY_NAMES.get(m, m)[:2]} Md {_fmt_v(md)} M {_fmt_v(mn)}")
        tick_labels.append("\n".join(parts))
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels)

    if n == 2:
        for g in range(len(group_labels)):
            ga = per_model_groups.get(models[0])
            gb = per_model_groups.get(models[1])
            if ga is None or gb is None:
                continue
            st = _paired_stats(ga[g], gb[g])
            if st:
                _sig_bracket(ax, g * step, g * step + 1,
                             f"{_fmt_p(st['p'])}, r = {st['r']:+.2f}",
                             level=0)


def _grouped_bar_panel(ax, per_model_values, models, category_labels, ylabel,
                       title, ylim=None, fmt="{:.2f}", paired_arrays=None):
    """Grouped bars: categories on x, one bar per model, values annotated.

    `paired_arrays` maps model -> list of per-sample arrays (one per category);
    when supplied and two models are present, each category gets a paired test.
    """
    n = len(models)
    width = 0.8 / max(n, 1)
    x = np.arange(len(category_labels), dtype=np.float64)
    drew = False
    for i, m in enumerate(models):
        vals = per_model_values.get(m)
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=np.float64)
        offs = x + (i - (n - 1) / 2.0) * width
        bars = ax.bar(offs, vals, width * 0.9, color=_model_color(m, i),
                      alpha=0.75, lw=0)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v),
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 1), textcoords="offset points",
                            ha="center", va="bottom", fontsize=STAT_FS)
        drew = True
    if not drew:
        _empty(ax)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=3)
    if ylim:
        ax.set_ylim(*ylim)

    if paired_arrays and n == 2:
        pa, pb = paired_arrays.get(models[0]), paired_arrays.get(models[1])
        if pa is not None and pb is not None:
            labels = []
            for c in range(len(category_labels)):
                st = _paired_stats(pa[c], pb[c])
                labels.append(_fmt_p(st["p"]) if st else "")
            y0, y1 = ax.get_ylim()
            for c, lab in enumerate(labels):
                if lab:
                    ax.text(x[c], y1 + (y1 - y0) * 0.02, lab, ha="center",
                            va="bottom", fontsize=STAT_FS)
            ax.set_ylim(y0, y1 + (y1 - y0) * 0.16)


def save_metrics_comparison_figure(per_model, out_base="metrics_comparison",
                                   logger=None):
    """Render the full evaluation-metric comparison.

    Parameters
    ----------
    per_model : dict
        model_name -> dict with any of the keys:
          'image_wise'    : {'RMSE': array, 'PSNR': array, 'SSIM': array}
          'spectral'      : DataFrame from compute_and_save_spectral_metrics
          'peak_errors'   : Errortable DataFrame from compute_and_save_peak_metrics
          'fit_status'    : fit-status DataFrame
          'bg'            : dict from compute_background_metrics
          'bg_per_image'  : its 'per_image' sub-dict, for paired tests
        Missing pieces leave their panel annotated "n/a" rather than failing.
    out_base : str
        Output path without extension; .png and .pdf are written.

    Returns
    -------
    str : path of the saved PNG.
    """
    preferred = [m for m in ("unet") if m in per_model]
    models = preferred + [m for m in per_model if m not in preferred]
    if not models:
        return None

    def imagewise(metric):
        return {m: per_model[m].get("image_wise", {}).get(metric, [])
                for m in models}

    def spectral(col):
        out = {}
        for m in models:
            df = per_model[m].get("spectral")
            out[m] = df[col].to_numpy() if df is not None and col in df else []
        return out

    def peakerr(col):
        """Keep NaNs in place so the two models stay index-aligned for pairing."""
        out = {}
        for m in models:
            df = per_model[m].get("peak_errors")
            out[m] = df[col].to_numpy() if df is not None and col in df else np.array([])
        return out

    with matplotlib.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(3, 4, figsize=(7.2, 7.4))
        letters = iter("abcdefghijkl")

        # --- Row 1: background reconstruction (vs GT background) ---
        _box_panel(axes[0, 0], imagewise("RMSE"), models, "RMSE",
                   "Background RMSE")
        _box_panel(axes[0, 1], imagewise("PSNR"), models, "PSNR (dB)",
                   "Background PSNR")
        _box_panel(axes[0, 2], imagewise("SSIM"), models, "SSIM",
                   "Background SSIM")

        bg_vals, bg_pairs = {}, {}
        for m in models:
            bg = per_model[m].get("bg") or {}
            if bg:
                bg_vals[m] = [abs(bg.get("bg_bias", np.nan)) * 1e3,
                              bg.get("leftover_in_bg", np.nan) * 1e3,
                              bg.get("contrast_deficit", np.nan) * 1e3]
            pi = per_model[m].get("bg_per_image")
            if pi:
                bg_pairs[m] = [np.abs(pi["bias"]), pi["leftover"],
                               pi["contrast_deficit"]]
        _grouped_bar_panel(axes[0, 3], bg_vals, models,
                           ["|bias|", "leftover", "$\\Delta$contrast"],
                           "value ($\\times 10^{-3}$)", "Background diagnostics",
                           fmt="{:.2f}", paired_arrays=bg_pairs or None)

        # --- Row 2: spectrum-level fidelity ---
        _box_panel(axes[1, 0], spectral("RMSE"), models, "RMSE",
                   "Spectrum RMSE")
        _box_panel(axes[1, 1], spectral("Pearson r"), models, "Pearson r",
                   "Spectrum Pearson r")
        _box_panel(axes[1, 2], spectral("Spearman rho"), models, "Spearman $\\rho$",
                   "Spectrum Spearman $\\rho$")
        _box_panel(axes[1, 3], spectral("Centroid % Error"), models, "error (%)",
                   "Centroid error")

        # --- Row 3: 2-Gaussian peak fitting ---
        rate_vals, rate_pairs = {}, {}
        for m in models:
            fs = per_model[m].get("fit_status")
            if fs is not None and len(fs):
                ok = (fs["pred_status"] == "ok").to_numpy()
                both = (fs["pred_first_valid"] & fs["pred_second_valid"]).to_numpy()
                usable = fs["pred_centroid_valid"].to_numpy()
                rate_vals[m] = [ok.mean() * 100, both.mean() * 100,
                                usable.mean() * 100]
                rate_pairs[m] = [ok, both, usable]

        if rate_vals:
            low = min(min(v) for v in rate_vals.values())
            ylim = (max(0.0, min(90.0, low - 2.0)), 100.9)
        else:
            ylim = None

        # Success is a paired binary outcome -> McNemar, not Wilcoxon.
        _grouped_bar_panel(axes[2, 0], rate_vals, models,
                           ["fit ok", "both\npeaks", "usable"],
                           "rate (%)", "Fit success", ylim=ylim, fmt="{:.1f}")
        if len(models) == 2 and len(rate_pairs) == 2:
            y0, y1 = axes[2, 0].get_ylim()
            for c in range(3):
                st = _mcnemar(rate_pairs[models[0]][c], rate_pairs[models[1]][c])
                if st:
                    axes[2, 0].text(c, y1 + (y1 - y0) * 0.02, _fmt_p(st["p"]),
                                    ha="center", va="bottom", fontsize=STAT_FS)
            axes[2, 0].set_ylim(y0, y1 + (y1 - y0) * 0.16)

        _grouped_box_panel(
            axes[2, 1],
            {m: [peakerr("firstPeaks_wavelengths_abs_errors")[m],
                 peakerr("secondPeaks_wavelengths_abs_errors")[m]] for m in models},
            models, ["1st peak", "2nd peak"],
            "|$\\Delta\\lambda$| (nm)", "Peak wavelength error")

        _grouped_box_panel(
            axes[2, 2],
            {m: [peakerr("firstPeakFWHM_abs_errors")[m],
                 peakerr("secondPeakFWHM_abs_errors")[m]] for m in models},
            models, ["1st peak", "2nd peak"],
            "|$\\Delta$FWHM| (nm)", "FWHM error")

        _box_panel(axes[2, 3], peakerr("PeakRatio_abs_errors"), models,
                   "|$\\Delta$ratio|", "Peak ratio error")

        for ax in axes.ravel():
            _panel_letter(ax, next(letters))

        handles = [plt.Rectangle((0, 0), 1, 1,
                                 facecolor=_model_color(m, i), alpha=0.75)
                   for i, m in enumerate(models)]
        fig.legend(handles, [DISPLAY_NAMES.get(m, m) for m in models],
                   loc="upper right", bbox_to_anchor=(0.99, 1.0),
                   ncol=len(models), frameon=False)

        n_ref = ""
        for m in models:
            fs = per_model[m].get("fit_status")
            if fs is not None:
                n_ref = f"n = {len(fs)} spectra.  "
                break
        fig.text(0.01, 0.010,
                 f"Box: median (line), IQR, 1.5$\\times$IQR whiskers; white diamond = mean "
                 f"(clipped when it falls outside the whisker range). Md = median, M = mean. {n_ref}\n"
                 f"P: two-sided Wilcoxon signed-rank on paired per-spectrum values; "
                 f"McNemar's exact test in panel i. r = matched-pairs rank-biserial effect size, "
                 f"+ve meaning the SA-SpecUNet value is larger.\n"
                 f"At n = 5000 nearly any systematic difference is significant, so judge "
                 f"practical magnitude by r and by the Md/M values, not by P alone.",
                 fontsize=4.6, va="bottom", ha="left", color="0.25", linespacing=1.5)

        fig.tight_layout(rect=[0, 0.055, 1, 0.965])

        png_path = f"{out_base}.png"
        fig.savefig(png_path, dpi=600)
        fig.savefig(f"{out_base}.pdf")
        plt.close(fig)

    if logger is not None:
        from .logger import log_print
        log_print(logger, f"[Figures] Metric comparison figure saved to "
                          f"{png_path} (+ .pdf)")
    return png_path
