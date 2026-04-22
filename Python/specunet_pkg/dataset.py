import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from scipy.io import loadmat
import sys

class SpecUNet_Dataset(Dataset):
    def __init__(self, X, Y=None, GTspt=None, spt=None):
        self.X = X
        self.Y = Y
        self.GTspt = GTspt
        self.spt = spt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float64)

        # For missing Y or GTspt, return zero tensors with expected shape
        if self.Y is None:
            y_tensor = torch.zeros_like(x_tensor)  # or another shape as needed
        else:
            y_tensor = torch.tensor(self.Y[idx], dtype=torch.float64)

        if self.GTspt is None:
            gt_tensor = torch.zeros_like(x_tensor)
        else:
            gt_tensor = torch.tensor(self.GTspt[idx], dtype=torch.float64)

        return x_tensor, y_tensor, gt_tensor

    def get_spt(self):
        return self.spt

def normalize_dataset(data, input_shape):
    """
    Normalizes dataset shape.
    - If input is image data (3D/4D): returns (N, 1, H, W).
    - If input is spectral data (2D): returns (N, Features).

    Parameters
    ----------
    data : np.ndarray
        Input dataset array (image or spectral curves).
    input_shape : tuple
        Expected single-sample image shape (e.g., (1, 128, 16)).
        Used primarily for image dimension validation.

    Returns
    -------
    np.ndarray
        Normalized dataset.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # --- CASE A: Handle 2D Data (Spectral Vectors) ---
    # Target: (N, 301)
    if data.ndim == 2:
        # Check if data is in (Features, Samples) format, e.g., (301, 5000)
        # We assume if dim0 is 301 (your spectral channels), it needs rotation.
        # OR if dim1 is significantly larger than dim0, it's likely (Feat, Samp).
        if data.shape[0] == 301 or (data.shape[1] > data.shape[0]):
            # print(f"2D Input detected {data.shape}: Transposing to (N, Features).")
            return data.T

        # Already (N, Features)
        return data

    # --- CASE B: Handle 3D/4D Data (Images) ---
    expected_c, expected_h, expected_w = input_shape

    # If array is 4D: (N, C, H, W) or mixed
    if data.ndim == 4:
        # Check if we need to swap H and W (e.g. 128 vs 16) inside the 4D array
        # This can happen if data is (N, C, W, H) instead of (N, C, H, W)
        _, _, h, w = data.shape
        if h == expected_w and w == expected_h:
            # print(f"4D Input detected {data.shape}: Swapping H and W axes.")
            data = np.swapaxes(data, 2, 3)
        return data

    # If array is 3D: Missing channel dimension
    if data.ndim == 3:
        # Get current dimensions
        dim0, dim1, dim2 = data.shape

        # Case 1: (N, H, W) -> Correct
        if dim1 == expected_h and dim2 == expected_w:
            pass

            # Case 2: (N, W, H) -> Transpose W and H
        elif dim1 == expected_w and dim2 == expected_h:
            # print(f"3D Input detected {data.shape}: Swapping spatial axes to (N, {expected_h}, {expected_w})")
            data = np.transpose(data, (0, 2, 1))

        # Case 3: (H, W, N) -> Move samples to front
        elif dim0 == expected_h and dim1 == expected_w:
            data = np.transpose(data, (2, 0, 1))

        # Case 4: (W, H, N) -> Swap H/W, Move samples to front
        elif dim0 == expected_w and dim1 == expected_h:
            data = np.transpose(data, (2, 1, 0))

        else:
            # Fallback: If dimensions don't match input_shape, try to guess N
            # This handles cases where input_shape might be (1,16,128) but data is slightly different
            pass

        # Add channel dimension (N, H, W) -> (N, 1, H, W)
        data = np.expand_dims(data, axis=1)
        return data

    raise ValueError(f"[dataset] Unsupported array dimensions: {data.ndim}")


def load_matlab_data(file_path, input_shape, input_name, target_name=None, gt_name=None, spt_name=None):
    """Loads MATLAB v5 or v7.3 data and expands dimensions."""

    # Detect file type from first bytes
    with open(file_path, 'rb') as f:
        header = f.read(8)

    sptimg4, tbg4, gt_spt, spt = None, None, None, None

    if header.startswith(b'MATLAB 5'):
        # MATLAB v5/v7.0 binary file
        try:
            mat_data = loadmat(file_path)
            sptimg4 = mat_data[input_name]
            if target_name is not None and target_name in mat_data:
                tbg4 = mat_data[target_name]
            if gt_name is not None and gt_name in mat_data:
                gt_spt = mat_data[gt_name]
            if spt_name is not None and spt_name in mat_data:
                spt = mat_data[spt_name]
        except FileNotFoundError:
            print(f"[dataset] Error: The file {file_path} was not found.")
            sys.exit(0)
        except KeyError as e:
            print(f"[dataset] Error: Required key {e} not found in the MATLAB file. Available dataset keys: {mat_data.keys()}")
            sptimg4 = None
            sys.exit(0)
        except Exception as e:
            print(f"[dataset] An unexpected error occurred: {e}")
            sys.exit(0)

    elif header.startswith(b'MATLAB 7'):
        # MATLAB v7.3 (HDF5) file
        with h5py.File(file_path, 'r') as f:
            try:
                sptimg4 = f[input_name][:]
                if target_name is not None and target_name in f:
                    tbg4 = f[target_name][:]
                if gt_name is not None and gt_name in f:
                    gt_spt = f[gt_name][:]
                if spt_name is not None and spt_name in f:
                    spt = f[spt_name][:]
            except FileNotFoundError:
                print(f"[dataset] Error: The dataset {file_path} was not found.")
            except KeyError as e:
                print(f"[dataset] Error: Required key {e} not found in the MATLAB file. Available dataset keys: {f.keys()}")
            except Exception as e:
                print(f"[dataset] An unexpected error occurred: {e}")

    else:
        try:
            with h5py.File(file_path, 'r') as f:
                # print("Dataset keys: ", f.keys())
                sptimg4 = f[input_name][:]
                if target_name is not None and target_name in f:
                    tbg4 = f[target_name][:]
                if gt_name is not None and gt_name in f:
                    gt_spt = f[gt_name][:]
                if spt_name is not None and spt_name in f:
                    spt = f[spt_name][:]
        except FileNotFoundError:
            print(f"[dataset] Error: The dataset {file_path} was not found.")
        except KeyError as e:
            print(f"[dataset] Error: Required key {e} not found in the MATLAB file. Available dataset keys: {f.keys()}")
        except Exception as e:
            print(f"[dataset] An unexpected error occurred: {e}")

    # Expand dims if loaded
    if sptimg4 is not None:
        # print(sptimg4.shape)
        # print(input_shape)
        sptimg4 = normalize_dataset(sptimg4, input_shape)
        # print(sptimg4.shape)
    if tbg4 is not None:
        # print(tbg4.shape)
        tbg4 = normalize_dataset(tbg4, tuple(input_shape))
        # print(tbg4.shape)
    if gt_spt is not None:
        # print(gt_spt.shape)
        gt_spt = normalize_dataset(gt_spt, tuple(input_shape))
        # print(gt_spt.shape)
    if spt is not None:
        # print(spt.shape)
        spt = normalize_dataset(spt, tuple(input_shape))
        # print(spt.shape)

    return sptimg4, tbg4, gt_spt, spt

def create_dataset(sptimg4, tbg4=None, gt_spt=None, spt=None):
    """Creates a SpecUNet_Dataset."""
    return SpecUNet_Dataset(sptimg4, tbg4, gt_spt, spt)

def create_dataloader(dataset, opt):
    """Creates a DataLoader based on training or testing."""
    return DataLoader(dataset, batch_size=opt['batch_size'], shuffle=opt['shuffle'],
                          num_workers=opt['num_workers'], pin_memory=opt['pin_memory'])


def get_train_datasets(opt, input_shape):
    """Main data loading script."""

    train_loader = None
    val_loader = None

    sptimg4_train, tbg4_train, GTspt_train, _ = load_matlab_data(
        file_path=opt['train']['args']['data_root'],
        input_name=opt['data_type']['input_name'],
        target_name=opt['data_type']['target_name'],
        gt_name=opt['data_type']['GTspt'],
        input_shape=input_shape
    )

    train_dataset = create_dataset(sptimg4_train, tbg4_train, GTspt_train)

    validation_split = opt['train']['dataloader']['validation_split']
    if validation_split > 0:
        val_size = int(validation_split * len(train_dataset))
        train_size = len(train_dataset) - val_size

        print(f"[dataset] "
            f"Validation split of {validation_split} used. Splitting selected training dataset into training and "
            f"validation datasets of sizes {train_size} and {val_size}, respectively.")

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = create_dataloader(train_dataset, opt['train']['dataloader']['args'])
        val_loader = create_dataloader(val_dataset, opt['train']['dataloader']['val_args'])

        print("[dataset] Loaded training and validation datasets!")

        return train_loader, val_loader

    else:
        sptimg4_test, tbg4_test, GTspt_test, _ = load_matlab_data(
            file_path=opt['test_sim']['args']['data_root'],
            input_name=opt['data_type']['input_name'],
            target_name=opt['data_type']['target_name'],
            gt_name=opt['data_type']['GTspt'],
            input_shape=input_shape
        )

        test_dataset = create_dataset(sptimg4_test, tbg4_test, GTspt_test)

        print(
            f"[dataset] No validation split passed in. Using testing dataset as validation dataset "
            f"for training and validation datasets of sizes {len(train_dataset)} and {len(test_dataset)}"
            f", respectively.")

        train_loader = create_dataloader(train_dataset, opt['train']['dataloader']['args'])
        val_loader = create_dataloader(test_dataset, opt['train']['dataloader']['val_args'])

        return train_loader, val_loader

def get_test_datasets(opt_dataset, input_shape, opt_phase):
     if opt_phase['test_sim']:
        sptimg4_test, tbg4_test, GTspt_test, spt_test = load_matlab_data(
            file_path=opt_dataset['test_sim']['args']['data_root'],
            input_name=opt_dataset['data_type']['input_name'],
            target_name=opt_dataset['data_type']['target_name'],
            gt_name=opt_dataset['data_type']['GTspt'],
            spt_name=opt_dataset['data_type']['spt'],
            input_shape=input_shape
        )
        test_dataset = create_dataset(sptimg4_test, tbg4_test, GTspt_test, spt_test)
        print(f"[dataset] Using test_sim dataset of size {len(test_dataset)}")
        return test_dataset

     else:
        sptimg4_test, tbg4_test, GTspt_test, _ = load_matlab_data(
            file_path=opt_dataset['test_exp']['args']['data_root'],
            input_name=opt_dataset['data_type']['input_name'],
            input_shape=input_shape)
        test_dataset = create_dataset(sptimg4_test, tbg4_test, GTspt_test)
        print(f"Using test_exp dataset of size {len(test_dataset)}")
        return test_dataset




