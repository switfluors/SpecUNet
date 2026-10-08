import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import mode
from nd2 import ND2File
import os
from PIL import Image
from scipy.io import savemat  # add once near imports
import imageio.v3 as iio  # pip install imageio
import hdf5storage  # <- writes MATLAB v7.3 (HDF5)



################ Function 1 ##################
# Load nd2 file and calibration information
# Display
##############################################
''' Input
1. select nd2 file
2. Calibration information:
    yi = [216, 230, 248, 266];
    xi = [436.6, 487.7, 546.5, 611.6];
    x0 = 90;
3. Optional Rotate the input image
'''

# ---- Choose  file ----
# Xiayi 532
yi = np.array([197, 211, 226, 246])  # detector pixel positions (Y) for known lines
x0 = 68

# # For Shahid Cell
# yi = np.array([179, 191, 205, 222])  # detector pixel positions (Y) for known lines
# x0 = 66

# # For Nile Red on Glass
# yi = np.array([208, 223, 239, 258])
# x0 = 81

# Set the ND2 path
nd2_path = r"D:\ExpData\Xiayi\561\561_08_002.nd2"

# # For Nile Red on Glass
# nd2_path = r"D:\HongjingMao\Manuscript\NileRed\8.07 Nile Red on glass\Channel561_narrow003.nd2"


# Calibration Information
xi = np.array([436.6, 487.7, 546.5, 611.6])  # known wavelengths (nm)
fy  = np.arange(500, 801, 3)   # OG 550:3:800
fy2 = np.arange(500, 801, 1)   # OG 550:1:800
fy3 = np.arange(500, 801, 3)   # for photon calc

# Linear fit: pixel (Y) vs wavelength
coeffs = np.polyfit(xi, yi, 1)
fx  = np.polyval(coeffs, fy)
fx2 = np.polyval(coeffs, fy2)
fx3 = np.polyval(coeffs, fy3)

print(f"Opening ND2: {nd2_path}")
with ND2File(nd2_path) as nd2f:
    sizes = nd2f.sizes
    T = sizes.get('T', 1)
    H = sizes['Y']
    W = sizes['X']

    A = np.zeros((H, W, T), dtype=np.float32)
    for i in range(T):
        A[:, :, i] = nd2f.read_frame(i)

A_rot = np.rot90(A, k=2)  # 180° rotation

########Optional Choose the rotate image as the input###########
A9 = A.copy()
# A9 = A_rot.copy()
################################################################

print("Loaded A with shape:", A.shape, "dtype:", A.dtype)
# Average across time
Ave1 = A9.mean(axis=2)
# %% If it's tiff file:
import tifffile as tiff

print(f"Opening TIFF: {nd2_path}")
arr = tiff.imread(nd2_path)

if arr.ndim == 2:
    A = arr.astype(np.float32)[..., None]                   # (H, W, 1)
elif arr.ndim == 3:
    # assume (T, H, W) -> move T to last
    A = np.moveaxis(arr, 0, 2).astype(np.float32)           # (H, W, T)
elif arr.ndim >= 4:
    # Try to find T and collapse others
    # Bring to (T, H, W) by moving the longest non-spatial axis to front
    Hdim, Wdim = arr.shape[-2], arr.shape[-1]
    core = arr
    # collapse any channel/Z dims by max
    while core.ndim > 3:
        # keep the first axis as T, reduce the next axis
        core = core.max(axis=1)
    A = np.moveaxis(core, 0, 2).astype(np.float32)          # (H, W, T)
else:
    raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

H, W, T = A.shape
# A_rot = np.rot90(A, k=2)
A9 = A.copy()
print("Loaded A with shape:", A.shape, "dtype:", A.dtype)

# Average across time
Ave1 = A9.mean(axis=2)
# %%
# For each row, find x position (within first 100 columns) of max intensity
slit_position = np.argmax(Ave1[:, :100], axis=1) + 1  # +1 to mimic MATLAB's 1-based intermediate
# Mode → most common slit column (still 1-based here)
slit_value, _ = mode(slit_position, keepdims=False)
slit = int(slit_value)  # 1-based

print("Estimated slit (1-based column index):", slit)

# Quick visualization
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(Ave1, cmap='gray')
axs[0].set_title("Average image (Ave1)")
axs[0].axis('off')

axs[1].imshow(A[:, :, 0], cmap='gray')
axs[1].set_title("Frame 1")
axs[1].axis('off')
plt.tight_layout()
plt.show()
# %%
################ Function 2 ##################
# Optional remove background with GBE method (If preparing training data for denoising, no need to remove background, set back = 0
##############################################
''' Input
1. Default not to do background removal
2. If choose to do background removel,set: col_lo, col_hi, a=w< ?
Note: col_lo is the lower end column of the selected ROI, vice versa for col_high; w is the mean value of the ROI, a is a boolean mask of "dark" frames
'''
back = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)

# If choose to not remove background, disable the following calculation
col_lo, col_hi = 150, 350
roi = A[:, col_lo:col_hi+1, :]      # shape: H × (col_hi-col_lo+1) × T
w = roi.mean(axis=1)                 # shape: H × T   (mean across selected columns)
a = (w < 100)                        # boolean mask of "dark" frames per row (H × T)
for n in range(A.shape[0]):
    good_frames = a[n]             # shape (T,)
    row_stack = A[n, :, :]         # W × T
    if good_frames.any():
        back[n, :] = row_stack[:, good_frames].mean(axis=1)   # (W,)
    else:
        # MATLAB: mean([]) -> NaN -> appears black in imagesc
        # Use np.nan for identical behavior, or 0.0 if you prefer solid black.
        back[n, :] = np.nan

# Background Preview
plt.figure(figsize=(6, 4))
plt.imshow(back, cmap='gray')
plt.title("Estimated background (per row)")
plt.axis('off')
plt.show()


# ---- Background subtraction (memory-safe) ----
# Use broadcasting without tiling, and process frames in chunks to avoid large temps.
# If back has NaNs (your code may set NaN when no dark frames), replace them with 0.
back_f32 = np.nan_to_num(back, nan=0.0).astype(np.float32, copy=False)
# Do the subtract in-place on A, in chunks over T. This keeps peak RAM ~ size(A).
chunk = 2000  # tune for your machine; 1000–5000 is typical
for i0 in range(0, T, chunk):
    i1 = min(i0 + chunk, T)
    A[:, :, i0:i1] -= back_f32[:, :, None]   # broadcast subtract (H,W,1) from (H,W,chunk)

if np.any(back):  # or np.nanmax(np.abs(back)) > 0
    back_f32 = np.nan_to_num(back, nan=0.0).astype(np.float32, copy=False)
    chunk = 2000
    for i0 in range(0, T, chunk):
        i1 = min(i0 + chunk, T)
        A[:, :, i0:i1] -= back_f32[:, :, None]
# Get Final A1 (no extra allocation)
A1 = A

# # ---- Background subtraction & variants  ----
# back_stack = np.repeat(back[:, :, None], A.shape[2], axis=2)
# # Get Final A1
# A1 = A - back_stack
# %%
################ Function 3 ##################
# Peak Detection and Spectra Extraction (with spt)
##############################################
''' Input
1. t (threshold for find_peaks)
2. row_window (e.g., (-8, +8), inclusive)
3. strip_offsets (rows to sum for 1-D spectrum, relative to peak row)
'''

def detect_targets(A1, slit_1b, t, band_half=3, x_search_right=10, min_sep=3):
    """
       Find emitters as peaks in a narrow band around the slit.
       Returns TR as 1-based (y, x, n) to match MATLAB.
    """
    H, W, T = A1.shape
    slit0 = slit_1b - 1
    TR = []
    for n in range(T):
        c0 = max(0, slit0 - band_half)
        c1 = min(W - 1, slit0 + band_half)
        srow = np.nansum(A1[:, c0:c1 + 1, n], axis=1)
        peaks, _ = find_peaks(srow, height=t, distance=min_sep)
        for y in peaks:
            if 3 < (y + 1) < (H - 3):
                r0 = max(0, y - 2)
                r1 = min(H - 1, y + 2)
                cmax = min(W - 1, slit0 + x_search_right)
                x = np.nansum(A1[r0:r1 + 1, :cmax + 1, n], axis=0).argmax()
                TR.append([y + 1, x + 1, n + 1])
    return np.array(TR, dtype=int).reshape(-1, 3)

def extract_spectra_and_patches(
    A, A1, TR, fx, fx2, fx3, x0, y0_shift=0,
    row_offsets=None,          # e.g., np.array([-8,-7,...,8]) or [-5,..,11]
    row_window=None,           # e.g., (-8, 8)  -> inclusive
    target_rows=16,            # final vertical size after resizing
    strip_offsets=tuple(range(-3, 4))  # rows to sum for 1-D spectrum
):
    """
    For each target (y,x,n) in TR (1-based):
      - Build spectral columns from fx/fx3 mapped to image columns
      - Sum rows given by strip_offsets to get raw 1-D spectra (on fx)
      - Interpolate 1-D spectrum onto fx2 (bs) and compute photons on fx3
      - Collect a patch from A1 using row_offsets/row_window and resize to (target_rows x 128)
      - ALSO: build the final 1×128 spectra per detection (spt) by resampling the raw 1-D spectrum to 128 bins

    Returns:
      final_bbimg : (target_rows x 128 x N) resized patches (from A1)
      bs          : (N x len(fx2)) spectra on integer-nm grid
      s_photons   : (N,) photon budgets (fx3 grid)
      spt_rows    : (#strip_rows, N) 1-based row indices actually used
      spt         : (N x 128) final 1×128 spectra from raw A using strip_offsets
    """
    H, W, T = A.shape
    TR = np.asarray(TR, dtype=int)
    strip_offsets = np.asarray(strip_offsets, dtype=int)

    # Early-out shape (for consistent returns)
    def _empty_return():
        return (
            np.zeros((target_rows, 128, 0)),
            np.zeros((0, fx2.size)),
            np.zeros(0),
            np.zeros((len(strip_offsets), 0), dtype=int),
            np.zeros((0, 128), dtype=np.float64),
        )

    if TR.size == 0:
        return _empty_return()

    # --- configure vertical offsets for patches ---
    if row_offsets is not None:
        row_offsets = np.asarray(row_offsets, dtype=int)
    elif row_window is not None:
        start, end = int(row_window[0]), int(row_window[1])
        if end < start:
            raise ValueError("row_window must satisfy end >= start")
        row_offsets = np.arange(start, end + 1, dtype=int)
    else:
        row_offsets = np.arange(-8, 8 + 1, dtype=int)  # default 17 rows

    # Precompute integer wavelength columns (still 1-based like MATLAB)
    cols_fx  = np.round(fx ).astype(int)
    cols_fx3 = np.round(fx3).astype(int)

    bbimg_list, bs_list, photons_list, keep = [], [], [], []
    spt_list_rows = []  # 1-based rows used for spectra (audit trail)
    spec_raw_list = []  # raw spectra on the fx grid (one per detection)

    for (y1b, x1b, n1b) in TR:
        n0 = n1b - 1

        # 0-based columns mapped from wavelength, shifted by target x
        cols  = np.clip(cols_fx  - x0 + x1b - 1, 0, W - 1)  # (len(fx),)
        cols3 = np.clip(cols_fx3 - x0 + x1b - 1, 0, W - 1)  # (len(fx3),)

        # Rows for spectral sum (strip), relative to peak row
        spt_rows0 = np.clip((y1b + y0_shift - 1) + strip_offsets, 0, H - 1)  # 0-based rows
        spt_list_rows.append(spt_rows0 + 1)  # store as 1-based

        # --- Raw 1-D spectra on fx / fx3 directly from A ---
        spec_raw   = A[spt_rows0[:, None], cols[None, :],  n0].sum(axis=0)  # (len(fx),)
        spec_raw3  = A[spt_rows0[:, None], cols3[None, :], n0].sum(axis=0)  # (len(fx3),)
        spec_raw_list.append(spec_raw)

        # Interpolate spec_raw (samples at integer-rounded fx) onto fx2 (integer nm)
        wl_int = np.round(fx).astype(int)
        u_wl, inv = np.unique(wl_int, return_inverse=True)
        acc = np.zeros(u_wl.size, dtype=spec_raw.dtype)
        cnt = np.zeros(u_wl.size, dtype=np.int32)
        np.add.at(acc, inv, spec_raw)
        np.add.at(cnt, inv, 1)
        spec_unique = acc / np.maximum(cnt, 1)
        bs_list.append(np.interp(fx2, u_wl, spec_unique))  # (len(fx2),)

        # Photon budget on fx3 grid (your scale)
        photons_list.append(spec_raw3.sum() * 1.1 / 0.95)

        # ---- Patch from A1 (rows_any × columns on fx) ----
        rows_any = np.clip((y1b + y0_shift - 1) + row_offsets, 0, H - 1)
        patch    = A1[rows_any[:, None], cols[None, :], n0]  # (#rows × len(fx))
        bbimg_list.append(patch)

        # Flatness filter (variance-based on top/bottom rows of patch)
        top_flat    = np.all(np.std(patch[0:3, :], axis=0) == 0)
        bottom_flat = np.all(np.std(patch[-3:, :], axis=0) == 0)
        keep.append(not (top_flat or bottom_flat))

    # Stack & filter kept slices
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        return _empty_return()

    bbimg      = np.stack(bbimg_list, axis=2)[:, :, keep]          # (#rows x len(fx) x N)
    bs         = np.stack(bs_list, axis=0)[keep]                   # (N x len(fx2))
    s_photons  = np.asarray(photons_list, dtype=float)[keep]       # (N,)
    spt_rows   = np.stack(spt_list_rows, axis=1)[:, keep]          # (#strip_rows, N)
    spec_raw   = np.stack(spec_raw_list, axis=0)[keep]             # (N x len(fx))

    # --- Build final 1×128 spectra (spt) from the raw spectra (on fx) ---
    C = spec_raw.shape[1]
    x_src = np.arange(C)
    x_dst = np.linspace(0, C - 1, 128)
    spt = np.empty((spec_raw.shape[0], 128), dtype=np.float64)
    for j in range(spec_raw.shape[0]):
        spt[j, :] = np.interp(x_dst, x_src, spec_raw[j, :])

    # --- Resize patches to (target_rows x 128) ---
    rows_in, Cfx, N = bbimg.shape
    new_cols = np.linspace(0, Cfx - 1, 128)
    new_rows = np.linspace(0, rows_in - 1, target_rows)
    final_bbimg = np.empty((target_rows, 128, N), dtype=bbimg.dtype)
    for k in range(N):
        tmp = np.empty((rows_in, 128), dtype=bbimg.dtype)
        for r in range(rows_in):
            tmp[r, :] = np.interp(new_cols, np.arange(Cfx), bbimg[r, :, k])
        for c in range(128):
            final_bbimg[:, c, k] = np.interp(new_rows, np.arange(rows_in), tmp[:, c])

    return final_bbimg, bs, s_photons, spt_rows, spt, bbimg

# 1) detect peaks
TR = detect_targets(A1, slit_1b=slit, t=1000)
print("Found peaks:", TR.shape[0])

# 2) extract spectra + patches (and spt)
strip_offsets_used = tuple(range(-3, 4))  # (-3..+3)
final_bbimg, bs, s_photons, spt_rows, spt, bbimg  = extract_spectra_and_patches(
    A, A1, TR, fx=fx, fx2=fx2, fx3=fx3, x0=x0, y0_shift=0,
    row_window=(-10, 6),                  # 17-row patch for bbimg
    strip_offsets=strip_offsets_used      # rows used to build the spectrum
)

print("final_bbimg:", final_bbimg.shape,
      "bbimg",bbimg.shape,
      "bs:", bs.shape,
      "photons:", s_photons.shape,
      "spt_rows:", spt_rows.shape,
      "spt:", spt.shape)  # (N x 128)
# %%
################ Function 4 ##################
# Live visualization of the final output
# Optional Save the gif
##############################################
''' Input
1.Save as gif(y/n)
2. if y, provide save path
'''

# Live visualize all frames from 0 to final_bbimg.shape[2]-1
imgs = final_bbimg  # shape: (H, W, N)
if imgs.ndim != 3 or imgs.shape[2] == 0:
    raise ValueError("final_bbimg must be (H, W, N) with N > 0")

H, W, N = imgs.shape

# Fix contrast once to avoid flicker
vmin = np.nanpercentile(imgs, 1)
vmax = np.nanpercentile(imgs, 99)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
    vmin = vmax = None  # fallback to autoscale

plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(imgs[:, :, 0], cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax)
title = ax.set_title(f"Frame 1/{N}")
ax.set_xlabel("spectral pixels")
ax.set_ylabel("rows")

for n in range(N):  # 0 .. N-1
    im.set_data(imgs[:, :, n])
    title.set_text(f"Frame {n+1}/{N}")
    fig.canvas.draw_idle()
    plt.pause(0.01)  # adjust playback speed

plt.ioff()
plt.show()

def save_stack_to_gif(imgs, save_path, fps=20, vmin=None, vmax=None, upsample=1):
    if imgs.ndim != 3 or imgs.shape[2] == 0:
        raise ValueError("imgs must be (H, W, N) with N > 0")

    # Fix contrast if not provided
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = np.nanpercentile(imgs, 1)
        vmax = np.nanpercentile(imgs, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(imgs)), float(np.nanmax(imgs))

    frames = []
    for n in range(imgs.shape[2]):
        frame = imgs[:, :, n]
        frame = np.nan_to_num(frame, nan=vmin)
        norm = (frame - vmin) / (vmax - vmin)
        u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        if upsample > 1:
            u8 = np.kron(u8, np.ones((upsample, upsample), dtype=np.uint8))  # simple resize
        frames.append(u8)

    iio.imwrite(save_path, frames, duration=1.0/fps, loop=0)  # loop=0 => infinite loop
    print(f"Saved GIF to: {save_path}")
# %%
save_stack_to_gif(final_bbimg, r"D:\ExpData\Xiayi\561\561_08_002\561_08_002_t1000.gif", fps=30, vmin=vmin, vmax=vmax, upsample=2)
# %%
################ Function 5 ##################
# Save the and optional normalize the output
# Optional save to the .mat file (final_bbimg and spt)
# optional save the spt as csv file
##############################################
def normalize_to_float64(img):
    img = img.astype(np.float64)
    img -= img.min()
    img /= img.max() + 1e-8
    return img

def save_tiff_image(img, save_path, normalize):
    if normalize:
        img = normalize_to_float64(img)
    img_uint16 = (img * 65535).clip(0, 65535).astype(np.uint16)
    Image.fromarray(img_uint16, mode='I;16').save(save_path)

def save_stack_as_individual_tiffs(final_bbimg, out_dir, prefix="frame", normalize=False, start_index=0):
    """
    Save each slice of final_bbimg (H, W, N) as a single TIFF.
    """
    if final_bbimg.ndim != 3 or final_bbimg.shape[2] == 0:
        raise ValueError("final_bbimg must be (H, W, N) with N > 0")

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(final_bbimg.shape[2]):
        fname = f"{prefix}_{i + start_index}.tif"   # no zero padding
        fpath = os.path.join(out_dir, fname)
        save_tiff_image(final_bbimg[:, :, i], fpath, normalize=normalize)
        paths.append(fpath)

    print(f"Saved {len(paths)} TIFF(s) to: {out_dir}")
    return paths

def save_v73_together(path, final_bbimg, spt, normalize_final=False, normalize_spt=False, compress=True):
    def maybe_norm(a, do=False):
        a = np.asarray(a)
        if do:
            a = a.astype(np.float64, copy=False)
            vmin, vmax = np.nanmin(a), np.nanmax(a)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                a = (a - vmin) / (vmax - vmin)
        return a

    data = {
        "final_bbimg": maybe_norm(final_bbimg, normalize_final),
        "spt":         maybe_norm(spt,         normalize_spt),
    }

    opts = hdf5storage.Options(
        matlab_compatible=True,
        store_python_metadata=False,
        compress=compress
    )

    hdf5storage.savemat(path, data, format="7.3", options=opts)
    print(f"Saved -v7.3 file with variables: {list(data.keys())} -> {path}")

# Example
save_v73_together(
    r"D:\ExpData\Xiayi\561\561_08_002\561_08_002_t1000_all.mat",
    final_bbimg,  # (H, W, N)
    spt,          # (N, 128) or (N, 301) etc.
    normalize_final=False,
    normalize_spt=False,
    compress=True
)
#%% Save all vars into one mat
def save_mat_stack(arr, save_path, var_name="final_bbimg", normalize=False, do_compression=True):
    """Save entire 3D stack to a .mat file."""
    out = arr.astype(np.float64, copy=False)
    if normalize:
        # per-volume min–max (not per-slice)
        vmin = np.nanmin(out)
        vmax = np.nanmax(out)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            out = (out - vmin) / (vmax - vmin)
    savemat(save_path, {var_name: out}, do_compression=do_compression)
    print(f"Saved MAT to: {save_path} (variable: '{var_name}')")

# Save each slice with per-slice normalization
save_stack_as_individual_tiffs(
    final_bbimg,
    out_dir=r"D:\ExpData\Xiayi\561\561_08_002",
    prefix="561_08_002_t1000",
    normalize=True,   # per-slice min–max
    start_index=1,
)

# Optional save as .mat file
save_mat_stack(
    final_bbimg,
    save_path=r"D:\ExpData\Xiayi\561\561_08_002\561_08_002_t1000.mat",
    var_name="final_bbimg",
    normalize=False,
    do_compression=True
)
# --- Save spt (N x 128) using the same helper ---
save_mat_stack(
    spt,
    save_path=r"D:\ExpData\Xiayi\561\561_08_002\561_08_002_t1000_spt.mat",
    var_name="spt",
    normalize=False,           # usually keep spectra unnormalized; change if you want
    do_compression=True
)

# Also save spt as CSV
# np.savetxt(r"D:\ExpData\Xiayi\561\561_08_002\561_08_002_t1000_spt.csv", spt, delimiter=",")
print("Saved spt (.mat via save_mat_stack and .csv)")
# %% Cropping the background
def background_crop(
    A, A1, fx, x0,
    row_window=(-8, 8),
    target_rows=16,
    num_patches=500,
    p99_thresh=110.0,        # reject if 99th percentile >= this
    max_thresh=110.0,       # reject if max >= this
    avoid_TR=None,          # optional: array of (y,x,n) 1-based from your detect_targets
    avoid_row_margin=6,     # rows to avoid near TR centers
    avoid_col_margin=10,    # columns to avoid near TR centers (approx in image cols)
    max_tries_factor=10,    # up to num_patches * factor tries
    rng=None                # optional np.random.Generator for reproducibility
):
    """
    Randomly sample background patches that follow your spectral mapping.
    Returns:
        bg_patches : (target_rows, 128, K) float array (same dtype as A1)
        centers    : (K, 3) int array of accepted (y, x, n) in 1-based indexing
    Notes:
      - Uses fx/x0 shift exactly like your spectral pipeline.
      - Resizes to (target_rows x 128) with the same 2-pass np.interp you use.
      - 'Too bright' crops (by max and 99th percentile) are rejected.
      - If avoid_TR is provided, candidates near any detected signal are skipped.
    """
    A = np.asarray(A)
    A1 = np.asarray(A1)
    fx = np.asarray(fx)
    H, W, T = A1.shape
    start, end = int(row_window[0]), int(row_window[1])
    row_offsets = np.arange(start, end + 1, dtype=int)

    # Precompute integer wavelength columns for a generic x=1 anchor (1-based inside math)
    cols_fx_base = np.round(fx).astype(int)

    # Simple RNG
    if rng is None:
        rng = np.random.default_rng()

    # Optional: build quick lookups for avoidance if TR is given
    has_avoid = (avoid_TR is not None) and (np.asarray(avoid_TR).size > 0)
    if has_avoid:
        TR = np.asarray(avoid_TR, dtype=int)
        TR_y = TR[:, 0]  # 1-based
        TR_x = TR[:, 1]  # 1-based
        TR_n = TR[:, 2]  # 1-based

    # Helpers to resize like your code (bilinear via two np.interp passes)
    def _resize_patch(patch_2d, out_rows=target_rows, out_cols=128):
        rows_in, cols_in = patch_2d.shape
        new_cols = np.linspace(0, cols_in - 1, out_cols)
        new_rows = np.linspace(0, rows_in - 1, out_rows)
        tmp = np.empty((rows_in, out_cols), dtype=patch_2d.dtype)
        for r in range(rows_in):
            tmp[r, :] = np.interp(new_cols, np.arange(cols_in), patch_2d[r, :])
        out = np.empty((out_rows, out_cols), dtype=patch_2d.dtype)
        for c in range(out_cols):
            out[:, c] = np.interp(new_rows, np.arange(rows_in), tmp[:, c])
        return out

    # Storage
    kept = []
    centers = []

    tries = 0
    max_tries = int(num_patches * max_tries_factor)

    while (len(kept) < num_patches) and (tries < max_tries):
        tries += 1

        # Sample a random frame (1-based n), y-center (1-based), and a spectral x-center (1-based)
        n1b = int(rng.integers(1, T + 1))
        # y must allow row_window inside [1..H]
        y_min = 1 - start
        y_max = H - end
        if y_min > y_max:
            # row_window invalid for image height; bail out
            break
        y1b = int(rng.integers(y_min, y_max + 1))

        # Any x works; clipping will handle edges (fine for background)
        x1b = int(rng.integers(1, W + 1))

        # Avoidance guard near detected signals
        if has_avoid:
            # same frame constraint (or allow near if different frames)
            same_frame = (TR_n == n1b)
            if np.any(same_frame):
                dy = np.abs(TR_y[same_frame] - y1b)
                dx = np.abs(TR_x[same_frame] - x1b)
                if np.any((dy <= avoid_row_margin) & (dx <= avoid_col_margin)):
                    continue  # too close to a signal; resample

        # Build spectral columns per your mapping (0-based inside arrays)
        cols = np.clip(cols_fx_base - x0 + x1b - 1, 0, W - 1)  # (len(fx),)
        rows_any = np.clip((y1b - 1) + row_offsets, 0, H - 1)

        # Extract the candidate patch (rows × len(fx)) from A1 in chosen frame (0-based n)
        n0 = n1b - 1
        patch = A1[rows_any[:, None], cols[None, :], n0]  # (#rows × len(fx))

        # Brightness rejection
        p99 = float(np.nanpercentile(patch, 99))
        mx = float(np.nanmax(patch))
        if not (np.isfinite(p99) and np.isfinite(mx)):
            continue
        if (p99 >= p99_thresh) or (mx >= max_thresh):
            continue

        # Resize to (target_rows x 128) like your final_bbimg
        patch_rs = _resize_patch(patch, out_rows=target_rows, out_cols=128)

        kept.append(patch_rs)
        centers.append((y1b, x1b, n1b))

    if len(kept) == 0:
        # empty return consistent with your other functions
        return np.zeros((target_rows, 128, 0), dtype=A1.dtype), np.zeros((0, 3), dtype=int)

    bg_patches = np.stack(kept, axis=2)                  # (target_rows, 128, K)
    centers = np.asarray(centers, dtype=int)             # (K, 3) 1-based (y,x,n)
    return bg_patches, centers

import numpy as np

# After you have A, A1, fx, x0, TR from your pipeline: crop background from single stack:
bg_patches, bg_centers = background_crop(
    A, A1, fx, x0,
    row_window=(-8, 8),
    target_rows=16,
    num_patches=500,
    p99_thresh=110.0,
    max_thresh=110.0,
    avoid_TR=TR,               # optional; pass None to ignore
    avoid_row_margin=6,
    avoid_col_margin=10
)
print("Background patches:", bg_patches.shape)  # (16, 128, K)
print("Centers (1-based y,x,n):", bg_centers.shape)

# %% Live visualize background patches (bg_patches: H x W x K)
imgs = bg_patches
if imgs.ndim != 3 or imgs.shape[2] == 0:
    raise ValueError("bg_patches must be (H, W, K) with K > 0")

H, W, K = imgs.shape

# Fix contrast once to avoid flicker
vmin = np.nanpercentile(imgs, 1)
vmax = np.nanpercentile(imgs, 99)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
    vmin = vmax = None  # fallback to autoscale

plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(imgs[:, :, 0], cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax)
title = ax.set_title(f"Background patch 1/{K}")
ax.set_xlabel("spectral pixels")
ax.set_ylabel("rows")

for k in range(K):
    im.set_data(imgs[:, :, k])
    title.set_text(f"Background patch {k+1}/{K}")
    fig.canvas.draw_idle()
    plt.pause(0.01)  # adjust playback speed

plt.ioff()
plt.show()
# %% gif:
# Optional: save a GIF of the background patches
save_stack_to_gif(
    bg_patches,
    r"D:\YunshuData\bg_patches.gif",
    fps=30, vmin=vmin, vmax=vmax, upsample=2
)

# %% Crop spectra based on localization: load hdf5 file from picasso and then process:
