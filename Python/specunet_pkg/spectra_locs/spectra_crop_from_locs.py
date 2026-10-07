# =========================
# SECTION 1. Load ND2 + calibrate + preview
# =========================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from nd2 import ND2File

# -------- User input --------
# Calibration information:
yi = np.array([166, 183, 200, 222], dtype=np.float32)
xi = np.array([436.6, 487.7, 546.5, 611.6], dtype=np.float32)
x0 = 30

# MATLAB-aligned wavelength / pixel grids
fy  = np.arange(600, 801, 50, dtype=np.float32)  # 600:50:800 for xticks
fy2 = np.arange(600, 801, 1,  dtype=np.float32)  # 600:1:800 for interpolation

# ND2 path
# nd2_path = r"D:\HongjingMao\DNA_PAINT\Cell001.nd2"
nd2_path = r"D:\HongjingMao\PreviousProject_and_Data\Nanosphere\Spectrum5LP519F12.nd2"

print(f"Opening ND2: {nd2_path}")
with ND2File(nd2_path) as nd2f:
    sizes = nd2f.sizes
    T = sizes.get('T', 1)
    H = sizes['Y']
    W = sizes['X']
    A = np.zeros((H, W, T), dtype=np.float32)
    for i in range(T):
        A[:, :, i] = nd2f.read_frame(i)

print("Loaded A:", A.shape, A.dtype)

# Optional rotate (choose the one you want to use)
A_rot = np.rot90(A, k=2)
# A_work = A.copy()
A_work = A_rot.copy()

# Average frame & slit finding
Ave1 = A_work.mean(axis=2)
slit_position = np.argmax(Ave1[:, :100], axis=1) + 1
slit = int(np.bincount(slit_position).argmax())
print("Estimated slit (1-based col):", slit)

# Quick preview
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(Ave1, cmap='gray'); axs[0].set_title("Average (raw)"); axs[0].axis('off')
axs[1].imshow(A_work[:, :, 0], cmap='gray'); axs[1].set_title("Frame 0"); axs[1].axis('off')
plt.tight_layout(); plt.show()

# Calibration fit: pixel row = a*lambda + b
coeffs = np.polyfit(xi, yi, 1).astype(np.float32)
fx  = np.polyval(coeffs, fy ).astype(np.float32)   # for xticks / span
fx2 = np.polyval(coeffs, fy2).astype(np.float32)   # pixel queries for interpolation
print("coeffs (row = a*λ + b):", coeffs)
print("fx pixel span:", float(np.abs(fx[-1] - fx[0])))

#%%
# =========================
# SECTION 2. Background estimation + subtraction
# =========================

# Choose columns that are dark to estimate row-wise background
col_lo, col_hi = 45, 48
roi = A_work[:, col_lo:col_hi+1, :]
# roi = A_work
w_mean = roi.mean(axis=1)                 # (H, T)
# dark_mask = (w_mean < 190)                # tweak threshold if needed, for cell
dark_mask = (w_mean < 106)                # tweak threshold if needed, for nanosphere


back = np.zeros((A_work.shape[0], A_work.shape[1]), dtype=np.float32)
# # If choose to not remove background, disable the following calculation
# for n in range(A_work.shape[0]):
#     good = dark_mask[n]
#     if good.any():
#         back[n, :] = A_work[n, :, :][:, good].mean(axis=1)
#     else:
#         back[n, :] = np.nan

# Visualize background
plt.figure(figsize=(6,4))
vmin, vmax = np.nanpercentile(back, [1, 99])
plt.imshow(back, cmap='gray', vmin=vmin, vmax=vmax); plt.title("Background (back)")
plt.axis('off'); plt.show()

# Subtract background in-place, chunked for RAM safety
back_f32 = np.nan_to_num(back, nan=0.0).astype(np.float32, copy=False)
A1 = A_work.copy()
chunk = 2000
for i0 in range(0, A1.shape[2], chunk):
    i1 = min(i0 + chunk, A1.shape[2])
    A1[:, :, i0:i1] -= back_f32[:, :, None]

print("A1 (bg-subtracted) shape:", A1.shape)

# Preview A1 with substracted background
plt.figure(figsize=(6, 4))
plt.imshow(A1.mean(axis=2), cmap='gray')  # Plot average image with background substraction
plt.title("Average clean image")
plt.axis('off')
plt.show()

# # (Optional) Save the results
# np.save("stack_bgsub.npy", A1.astype(np.float32))
# np.save("background_map.npy", back.astype(np.float32))

# After this section we will  have:
# - A1 :  cleaned movie, shape (H,W,T)
# - back : the static background 2D image, shape (H,W)
#%%
# =========================
# SECTION 3. Helpers
# =========================
import pandas as pd
import h5py
from scipy.interpolate import PchipInterpolator
has_pchip = True
from dataclasses import dataclass, field
from typing import Optional
import hdf5storage
import os
import imageio.v2 as imageio
import io


def pchip_interp(x, y, x_new):
    if has_pchip:
        f = PchipInterpolator(x, y, extrapolate=True)
        return f(x_new)
    return np.interp(x_new, x, y, left=0.0, right=0.0)

def load_locs(path: str) -> np.ndarray:
    """
    Return TR with columns [x_pix, y_pix, frame, photons].
    Accepts HDF5 with '/locs' (fields: frame,x,y,photons) or CSV with x,y,frame,photons.
    """
    if path.lower().endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as f:
            if "locs" not in f:
                raise KeyError("HDF5 missing '/locs'")
            arr = f["/locs"][...]
            names = arr.dtype.names
            req = {"frame", "x", "y", "photons"}
            if not names or not req.issubset(set(names)):
                raise KeyError(f"/locs fields missing. Found: {names}")
            TR = np.zeros((arr.shape[0], 4), dtype=np.float32)
            TR[:, 0] = arr["x"].astype(np.float32)
            TR[:, 1] = arr["y"].astype(np.float32)
            TR[:, 2] = arr["frame"].astype(np.float32)
            TR[:, 3] = arr["photons"].astype(np.float32)
            return TR

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for k in ["x", "y", "frame", "photons"]:
        if k not in cols:
            raise KeyError(f"CSV missing column: {k}")
    TR = np.zeros((len(df), 4), dtype=np.float32)
    TR[:, 0] = pd.to_numeric(df[cols["x"]], errors="coerce").values.astype(np.float32)
    TR[:, 1] = pd.to_numeric(df[cols["y"]], errors="coerce").values.astype(np.float32)
    TR[:, 2] = pd.to_numeric(df[cols["frame"]], errors="coerce").values.astype(np.float32)
    TR[:, 3] = pd.to_numeric(df[cols["photons"]], errors="coerce").values.astype(np.float32)
    return TR

@dataclass
class Params:
    yi: np.ndarray
    xi: np.ndarray
    x0: int = 30
    fy:  np.ndarray = field(default_factory=lambda: np.arange(600, 801, 100, dtype=np.float32))
    fy2: np.ndarray = field(default_factory=lambda: np.arange(600, 801, 1,  dtype=np.float32))
    y0: int = 0
    bw: int = 2
    w:  int = 8
    live: int = 1            # 1 = show live plot, 0 = no window
    frame_to_plot: Optional[int] = None


# Noramlization 
def normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1], ignoring NaNs. If flat, returns zeros."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    vmin = arr[finite].min()
    vmax = arr[finite].max()
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return out

# Save the output to .mat -v7.3 file
def save_results_to_mat_v73(mat_path: str,
                            as_prof: np.ndarray,
                            bs_spec: np.ndarray,
                            specimage_interp: np.ndarray,
                            bb3: np.ndarray,
                            TR: np.ndarray,
                            params: Params,
                            fx: np.ndarray,
                            fx2: np.ndarray,
                            normalize: bool = False):
    """
    Save analysis outputs to a MATLAB -v7.3 .mat file.
    """
    os.makedirs(os.path.dirname(mat_path), exist_ok=True)
    if normalize:
        as_save   = normalize_01(as_prof)
        bs_save   = normalize_01(bs_spec)
        spec_save = normalize_01(specimage_interp)
        bb3_save  = normalize_01(bb3)
    else:
        as_save   = as_prof
        bs_save   = bs_spec
        spec_save = specimage_interp
        bb3_save  = bb3
    mdict = {
        'as_prof': as_save,                # (N, 2w+1)
        'bs_spec': bs_save,                # (N, len(fy2))
        'specimage_interp': spec_save,     # (N, 2w+1, len(fy2))
        'bb3': bb3_save,                   # (1, spec_len, N)
        'TR': TR,                          # raw localization table
        'fy': params.fy,
        'fy2': params.fy2,
        'xi': params.xi,
        'yi': params.yi,
        'fx': fx,                          # pixel positions for fy
        'fx2': fx2,                        # pixel positions for fy2
        'x0': params.x0,
        'y0': params.y0,
        'bw': np.int32(params.bw),
        'w':  np.int32(params.w),
    }
    # This writes a genuine -v7.3 MAT file (HDF5)
    hdf5storage.savemat(
        mat_path,
        mdict,
        format='7.3',
        oned_as='row'
    )
    print(f"Saved v7.3 .mat file to:\n  {mat_path}")


# save spectra image to tiffs
def save_specimages_as_tiffs(specimage_interp: np.ndarray,
                             out_dir: str,
                             normalize: bool = False,
                             basename: str = "spec"):
    """
    Save each specimage_interp[n, :, :] as its own TIFF file.
    No stacking, just separate files.
    """
    os.makedirs(out_dir, exist_ok=True)
    N = specimage_interp.shape[0]

    for i in range(N):
        img = specimage_interp[i, :, :]

        if normalize:
            img_norm = normalize_01(img)
            img_u16 = (img_norm * 65535).astype(np.uint16)
        else:
            # simple float->uint16 mapping: shift to >=0 and scale per image
            img = np.nan_to_num(img, nan=0.0)
            vmin = img.min()
            img = img - vmin
            vmax = img.max()
            if vmax > 0:
                img_u16 = (img / vmax * 65535).astype(np.uint16)
            else:
                img_u16 = np.zeros_like(img, dtype=np.uint16)

        fname = os.path.join(out_dir, f"{basename}_{i:05d}.tif")
        imageio.imwrite(fname, img_u16)

    print(f"Saved {N} spectral TIFFs to:\n  {out_dir}")
#%%
# =========================
# SECTION 4. Spectral cutout + interpolation
# =========================
def spectral_cutout_and_interpolate(A_stack: np.ndarray,
                                    back_map: np.ndarray,
                                    TR: np.ndarray,
                                    params: Params,
                                    coeffs_for_dispersion: np.ndarray):
    """
    Implements MATLAB Section 7 behavior (updated):
      - fy  = 600:50:800
      - fy2 = 600:800
      - s = TR[:, [2,1,3,4]] with xc applied to y (to mimic your MATLAB line)
      - spec columns = round(fx(1)):round(fx(end)) shifted by (-x0 + xC)
      - as_prof shape = (nn, 7) : mean across row band -> 1x7
      - bs_spec and specimage_interp interpolate from pixel coordinates (spec_cols) to fx2
      - bb3 sums across the row band -> (1, spec_len, nn)
    """
    H, W, F = A_stack.shape
    print("DEBUG: A_stack shape:", A_stack.shape)
    print("DEBUG: TR shape:", TR.shape)

    # Build s like MATLAB: TR[:, [2,1,3,4]] -> [y, x, frame, photons]
    s = np.empty_like(TR)
    s[:, 0] = TR[:, 1]    # y
    s[:, 1] = TR[:, 0]    # x
    s[:, 2] = TR[:, 2]    # frame (assume 0-based; update if needed)
    s[:, 3] = TR[:, 3]    # photons
    s[:, 0] = s[:, 0]
    # If intended xc to shift x, comment the line above and use:
    # xc = integer
    # s[:, 1] = s[:, 1] + xc

    # If you want to restrict to one frame (like the MATLAB commented example), filter here:
    if params.frame_to_plot is not None:
        s = s[s[:, 2].astype(int) == int(params.frame_to_plot)]
    print("DEBUG: s shape after frame filter:", s.shape)

    # Predict pixel locations for fy and fy2 (already computed in Section 1, but rebuild here for safety)
    fx  = np.polyval(coeffs_for_dispersion, params.fy ).astype(np.int64)
    fx2 = np.polyval(coeffs_for_dispersion, params.fy2).astype(np.float32)

    # Spectral pixel columns: round(fx(1)):round(fx(end))
    spec_cols = np.arange(int(np.round(fx[0])), int(np.round(fx[-1])) + 1, dtype=np.int64)
    spec_len  = spec_cols.size

    # Preallocate outputs
    nn = s.shape[0]
    row_len = 2*params.w + 1                 # 7 when w=3
    bs_len  = params.fy2.size                # 201 for 600..800

    as_prof          = np.zeros((nn, row_len), dtype=np.float32)         # (nn,7)
    bs_spec          = np.zeros((nn, bs_len),  dtype=np.float32)         # (nn,201)
    specimage_interp = np.zeros((nn, row_len, bs_len), dtype=np.float32) # (nn,7,201)
    bb3              = np.zeros((1, spec_len, nn), dtype=np.float32)     # (1,width,nn)

    # Helpers
    def clamp_idx(v, lo, hi):
        v = np.asarray(v, dtype=np.int64)
        v[v < lo] = lo
        v[v > hi] = hi
        return v

    # # Progress / live plotting
    # do_plot = bool(params.live == 1 and plt is not None)
    # if do_plot:
    #     plt.figure(figsize=(10, 6))
    do_plot = bool(params.live == 1 and plt is not None)
    if do_plot:
        plt.ion()
        plt.figure(1, figsize=(10, 6))

    for n in range(nn):
        yC = int(round(s[n, 0]))
        xC = int(round(s[n, 1]))
        fr = int(round(s[n, 2]))

        # Assume frames are 0-based; adjust if your TR is 1-based.
        if fr < 0 or fr >= F:
            continue

        # Spatial indices for blink ROI
        yIdx = clamp_idx(np.arange(-params.w, params.w+1) + yC, 1, H) - 1
        xIdx = clamp_idx(np.arange(-params.w, params.w+1) + xC, 1, W) - 1

        # Spectral rows (shifted by y0)
        ySpec = clamp_idx(np.arange(-params.w, params.w+1) + yC + params.y0, 1, H) - 1

        # Spectral columns: (round(fx(1)):round(fx(end))) - x0 + xC
        specCols = spec_cols - params.x0 + xC
        specCols = clamp_idx(specCols, 1, W) - 1  # 0-based

        # Extract patches
        aa = A_stack[yIdx[:, None],   xIdx[None, :],   fr]
        bb = A_stack[ySpec[:, None],  specCols[None, :], fr]

        # Background subtraction (use 2D back_map; if you already subtracted, pass zeros)
        if back_map.ndim == 3 and back_map.shape[-1] == F:
            aa = aa - back_map[yIdx[:, None],  xIdx[None, :], fr]
            bb = bb - back_map[ySpec[:, None], specCols[None, :], fr]
        else:
            aa = aa - back_map[yIdx[:, None],  xIdx[None, :]]
            bb = bb - back_map[ySpec[:, None], specCols[None, :]]

        # Row band for averaging (w±bw), in 0-based index inside the 7 rows
        rowBand = np.arange(params.w - params.bw, params.w + params.bw + 1)
        rowBand = rowBand[(0 <= rowBand) & (rowBand < aa.shape[0])]

        # as_prof: mean across the selected rows -> 1 x 7 (mean over rows, i.e., axis=0)
        as_prof[n, :] = aa[rowBand, :].mean(axis=0).astype(np.float32)

        # Spectrum line: average across rowBand -> 1D length spec_len
        spec_line = bb[rowBand, :].mean(axis=0).astype(np.float32)

        # Interpolate from pixel coords to fx2 (pixel units)
        x_src   = spec_cols.astype(np.float32)
        x_query = fx2.astype(np.float32)
        bs_spec[n, :] = pchip_interp(x_src, spec_line, x_query).astype(np.float32)
        # bs_spec[n, :] = np.interp(x_query, x_src, spec_line).astype(np.float32)

        # Full 2D spectral interpolation per row
        rows_interp = []
        for r in range(bb.shape[0]):
            rows_interp.append(pchip_interp(x_src, bb[r, :].astype(np.float32), x_query))
        specimage_interp[n, :, :] = np.vstack(rows_interp).astype(np.float32)

        # bb3: sum over the rowBand → length spec_len
        bb3[0, :, n] = bb[rowBand, :].sum(axis=0).astype(np.float32)

        # Progress 10x
        if n % max(1, nn // 10) == 0:
            pct = int(round(100.0 * n / max(1, nn-1)))
            print(f"Progress: {pct} %")

        # # Live display for several stacks: (optional)
        # if do_plot and (n < 20) and (n % 5 == 0):
        #     plt.clf()
        #     ax1 = plt.axes([0.06, 0.55, 0.17, 0.4]); ax1.imshow(aa, aspect='equal'); ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("Blink ROI")
        #     ax2 = plt.axes([0.30, 0.55, 0.65, 0.4]); ax2.imshow(bb, aspect='equal', cmap='gray')
        #     xt = np.linspace(0, bb.shape[1]-1, len(params.fy)); ax2.set_xticks(xt); ax2.set_xticklabels([f"{v:.0f}" for v in params.fy]); ax2.set_yticks([]); ax2.set_title("Spectral ROI")
        #     ax3 = plt.axes([0.06, 0.10, 0.17, 0.35]); ax3.plot(as_prof[n, :], 'k-'); ax3.set_xlim(0, row_len-1); ax3.set_title("Blink profile")
        #     ax4 = plt.axes([0.30, 0.10, 0.65, 0.35]); ax4.plot(bs_spec[n, :], 'k-'); ax4.set_xlim(0, bs_len-1)
        #     xt2 = np.linspace(0, bs_len-1, len(params.fy)); ax4.set_xticks(xt2); ax4.set_xticklabels([f"{v:.0f}" for v in params.fy])
        #     ax4.set_title("Spectrum (interp)"); ax4.set_xlabel("λ index"); ax4.set_ylabel("a.u.")
        #     plt.pause(0.001)

        # Live display: single window, updated every localization
        if do_plot:
            plt.figure(1)
            plt.clf()

            # Blink ROI
            ax1 = plt.axes([0.06, 0.55, 0.17, 0.4])
            ax1.imshow(aa, aspect='equal', cmap='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title("Blink ROI")

            # Spectral ROI
            ax2 = plt.axes([0.30, 0.55, 0.65, 0.4])
            ax2.imshow(bb, aspect='equal', cmap='gray')
            xt = np.linspace(0, bb.shape[1] - 1, len(params.fy))
            ax2.set_xticks(xt)
            ax2.set_xticklabels([f"{v:.0f}" for v in params.fy])
            ax2.set_yticks([])
            ax2.set_title("Spectral ROI")

            # Blink profile
            ax3 = plt.axes([0.06, 0.10, 0.17, 0.35])
            ax3.plot(as_prof[n, :], 'k-')
            ax3.set_xlim(0, row_len - 1)
            ax3.set_title("Blink profile")

            # Spectrum (interp)
            ax4 = plt.axes([0.30, 0.10, 0.65, 0.35])
            ax4.plot(bs_spec[n, :], 'k-')
            ax4.set_xlim(0, bs_len - 1)
            xt2 = np.linspace(0, bs_len - 1, len(params.fy))
            ax4.set_xticks(xt2)
            ax4.set_xticklabels([f"{v:.0f}" for v in params.fy])
            ax4.set_title("Spectrum (interp)")
            ax4.set_xlabel("λ index")
            ax4.set_ylabel("a.u.")

            plt.pause(0.001)

    if do_plot:
        plt.ioff()
        plt.close(1)

    return as_prof, bs_spec, specimage_interp, bb3


# ======= RUN =======
locs_path = r"D:\HongjingMao\PreviousProject_and_Data\Nanosphere\ForSpectraCrop\Spectrum5LP519F12_locs.hdf5"
TR = load_locs(locs_path)

# If TR frames are 0-based and your stack uses 0-based, do nothing.
# If TR frames are 1-based, shift: TR[:,2] -= 1

P = Params(
    yi=yi, xi=xi, x0=x0,
    fy=fy, fy2=fy2,
    y0=0, bw=2, w=8,
    live=0,                    # 0 = no live window, 1 = show
)

print("Processing…")
as_prof, bs_spec, specimage_interp, bb3 = spectral_cutout_and_interpolate(
    A_stack=A1,
    back_map=np.nan_to_num(back, nan=0.0).astype(np.float32, copy=False),
    TR=TR,
    params=P,
    coeffs_for_dispersion=coeffs
)

print("Results:")
print("as_prof:", as_prof.shape)
print("bs_spec:", bs_spec.shape)
print("specimage_interp:", specimage_interp.shape)
print("bb3:", bb3.shape)
# %% save the results
# ----- Define output paths -----
base_out_dir   = r"D:\HongjingMao\PreviousProject_and_Data\Nanosphere\ForSpectraCrop\spectrum5LP519F12"
mat_out_path   = os.path.join(base_out_dir, "Spectrum5LP519F12_results_v73.mat")
tiff_out_dir   = os.path.join(base_out_dir, "Spectrum5LP519F12_spec_tiffs")

# Optional: compute fx, fx2 again for saving
fx  = np.polyval(coeffs, fy ).astype(np.float32)
fx2 = np.polyval(coeffs, fy2).astype(np.float32)

# ----- Choose whether to normalize -----
normalize_all = False  # set True if you want [0,1] normalization

# 1) save everything to v7.3 MAT file
save_results_to_mat_v73(
    mat_path=mat_out_path,
    as_prof=as_prof,
    bs_spec=bs_spec,
    specimage_interp=specimage_interp,
    bb3=bb3,
    TR=TR,
    params=P,
    fx=fx,
    fx2=fx2,
    normalize=normalize_all
)

# 2) save each interpolated spectral image as a separate TIFF
save_specimages_as_tiffs(
    specimage_interp=specimage_interp,
    out_dir=tiff_out_dir,
    normalize=normalize_all,
    basename="specInterp"
)




# %%   Check localizations: (Sanity check)
locs_path = r"D:\HongjingMao\PreviousProject_and_Data\Nanosphere\ForSpectraCrop\Spectrum5LP519F12_locs.hdf5"
TR = load_locs(locs_path)

# sanity check for localization file
print("TR shape:", TR.shape)
if TR.shape[0] == 0:
    raise RuntimeError("No localizations found in TR – check your locs file.")

# build the 's' array for inspection (same as inside the function)
s = np.empty_like(TR)
s[:, 0] = TR[:, 1]    # y
s[:, 1] = TR[:, 0]    # x
s[:, 2] = TR[:, 2]    # frame
s[:, 3] = TR[:, 3]    # photons

# ---------- STEP 1: pick brightest localization and inspect ----------
import matplotlib.pyplot as plt
import numpy as np

idx_sorted = np.argsort(TR[:, 3])[::-1]  # sort by intensity
n0 = idx_sorted[0]
x0_loc = s[n0, 1]
y0_loc = s[n0, 0]
f0_tr  = s[n0, 2]
f0_img = int(f0_tr)  # adjust by +frame_offset if your TR is 0-based

print(f"Brightest loc index {n0} at (x={x0_loc:.2f}, y={y0_loc:.2f}, frame(TR)={f0_tr}, frame(A)={f0_img})")

plt.figure(figsize=(6, 5))
plt.imshow(A_work[:, :, f0_img], cmap='gray', aspect='equal')
plt.plot(x0_loc, y0_loc, 'r+', markersize=12, markeredgewidth=1.5)
plt.title(f"Full frame {f0_img} with brightest localization overlay")
plt.axis('image')
plt.show()

# visualize all localizations on a given frame
f_test = 35
mask_f = (s[:, 2].astype(int) == f_test)

plt.figure(figsize=(6, 5))
plt.imshow(A_work[:, :, f_test], cmap='gray', aspect='equal')
plt.plot(s[mask_f, 1], s[mask_f, 0], 'r+', markersize=6)
plt.title(f"All localizations in frame {f_test} (check alignment)")
plt.axis('image')
plt.show()

# ---------- after sanity check, continue with spectral cropping ----------
P = Params(yi=yi, xi=xi, x0=x0, fy=fy, fy2=fy2,y0=0, bw=2, w=8, live=1, frame_to_plot=None)

print("Processing…")
as_prof, bs_spec, specimage_interp, bb3 = spectral_cutout_and_interpolate(
    A_stack=A1,
    back_map=np.nan_to_num(back, nan=0.0).astype(np.float32, copy=False),
    TR=TR,
    params=P,
    coeffs_for_dispersion=coeffs
)







# %% check localization:
import h5py
import numpy as np
locs_path = r"D:\HongjingMao\PreviousProject_and_Data\Nanosphere\ForSpectraCrop\Spectrum5LP519F12_locs.hdf5"

with h5py.File(locs_path, "r") as f:
    print("HDF5 keys:", list(f.keys()))
    if "locs" in f:
        arr = f["locs"]
        print("locs shape:", arr.shape)
        print("locs dtype:", arr.dtype)
    else:
        print("No '/locs' dataset in this file!")
# %%   Other hdf5 file analyze
import os
import numpy as np
import tifffile as tiff
import hdf5storage

# data = np.load("spectral_results.npz", allow_pickle=True)
# as_prof = data["as_prof"]
# bs_spec = data["bs_spec"]
# specimage_interp = data["specimage_interp"]
# bb3 = data["bb3"]
#
# print("✅ Reloaded from NPZ:")
# print("as_prof:", as_prof.shape)
# print("bs_spec:", bs_spec.shape)
# print("specimage_interp:", specimage_interp.shape)
# print("bb3:", bb3.shape)

# ==== User-defined paths ====
out_dir_tiff = r"D:\HongjingMao\DNA_PAINT\spec_images"  # folder for individual TIFFs
mat_out_path = r"D:\HongjingMao\DNA_PAINT\spectral_results_v73.mat"

# Ensure output directory exists
os.makedirs(out_dir_tiff, exist_ok=True)
os.makedirs(os.path.dirname(mat_out_path), exist_ok=True)

# # ================================================
# # 1. Save each specimage_interp[i] as individual TIFF
# # ================================================
# N = specimage_interp.shape[0]
# print(f"Saving {N} individual TIFFs to:\n{out_dir_tiff}")
#
# # Optional: choose a stride to limit how many files to write for testing
# stride = 1  # e.g. set stride=100 to save every 100th localization
#
# for i in range(0, N, stride):
#     img = specimage_interp[i].astype(np.float32)
#     out_path = os.path.join(out_dir_tiff, f"spec_{i:07d}.tif")
#     tiff.imwrite(out_path, img, photometric='minisblack')
#
#     if i % max(1, N // 10) == 0:
#         print(f"Progress: {100 * i / N:.0f}% ({i}/{N})")
#
# print("✅ Finished writing individual TIFFs.")

# ================================================
# 2. Save all variables to MATLAB .mat (v7.3)
# ================================================
print(f"Saving all results to:\n{mat_out_path}")

hdf5storage.savemat(
    mat_out_path,
    {
        'as_prof': as_prof.astype(np.float32),
        'bs_spec': bs_spec.astype(np.float32),
        'specimage_interp': specimage_interp.astype(np.float32),
        'bb3': bb3.astype(np.float32),
    },
    format='7.3',
    store_python_metadata=False
)

print("✅ Saved all variables to MATLAB v7.3 file successfully.")

#%%
import h5py
import pandas as pd

# Path to your HDF5 file
h5_path = r"D:\Vignes\SharedDocuments\PicassoExamples\Well_4_1_001_bs3_with_sigma_uncertainty.hdf5"

# Open the file
with h5py.File(h5_path, "r") as f:
    # List all groups
    print("Groups:", list(f.keys()))

    # Access the 'localizations' group
    locs = f["localizations"]

    # Show available datasets
    print("Datasets:", list(locs.keys()))

    # Read each dataset into a DataFrame
    data = {key: locs[key][:] for key in locs.keys()}
    df = pd.DataFrame(data)

    # Print summary info
    print(df.head())
    print("\nPixel size (nm):", locs.attrs.get("pixel_size_nm", "Not found"))
    print("Description:", locs.attrs.get("description", "Not found"))

# You can now use df for analysis
print("\nDataFrame shape:", df.shape)
# %%
import h5py
import pandas as pd
import numpy as np
import os

# -------------------------------
# 1. Read existing HDF5 file
# -------------------------------
h5_path = r"D:\Vignes\SharedDocuments\PicassoExamples\Well_4_1_001_bs3_with_sigma_uncertainty.hdf5"

with h5py.File(h5_path, "r") as f:
    print("Groups:", list(f.keys()))
    locs = f["localizations"]
    print("Datasets:", list(locs.keys()))

    data = {key: locs[key][:] for key in locs.keys()}
    df = pd.DataFrame(data)

    print(df.head())
    print("\nPixel size (nm):", locs.attrs.get("pixel_size_nm", "Not found"))
    print("Description:", locs.attrs.get("description", "Not found"))

print("\nDataFrame shape:", df.shape)

# -------------------------------
# 2. Create Picasso-style HDF5 with sigma and uncertainty
# -------------------------------
out_h5 = os.path.splitext(h5_path)[0] + "_picasso_with_sigma_unc.hdf5"

# Required columns for Picasso (plus extra)
cols = [
    "frame", "x", "y", "photons", "sx", "sy", "bg",
    "lpx", "lpy", "net_gradient", "likelihood", "iterations",
    "sigma", "uncertainty"   # ← extra columns
]

# Check that all columns exist
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in the HDF5 file: {missing}")

# Make structured array
dtype = [(c, "f4") for c in cols]  # 32-bit floats
locs_arr = np.zeros(len(df), dtype=dtype)
for c in cols:
    locs_arr[c] = df[c].astype("float32")

# -------------------------------
# 3. Write to new HDF5
# -------------------------------
with h5py.File(out_h5, "w") as f:
    f.create_dataset("locs", data=locs_arr)
    f.attrs["pixelsize"] = 160.0     # nm/px
    f.attrs["generated_by"] = "Conversion script with sigma and uncertainty"

print(f"\n✅ Picasso-compatible HDF5 saved to:\n{out_h5}")
