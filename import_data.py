import numpy as np
import pandas as pd
import torch


##### USER-DEFINED FUNCTIONS #####
def f_get_norm_params(X, norm_mode):
    """
    Fit per-feature normalization parameters from X.

    To avoid leakage, fit these on the TRAINING split only, then apply them to
    train/val/test with f_apply_Normalization. (The upstream chl8856/DeepHit
    code normalizes the full dataset before splitting; this port fits on train.)

    norm_mode: 'standard' (zero mean, unit variance) or 'normal' (min-max).
    """
    if norm_mode == "standard":
        return {"mode": "standard", "mu": np.mean(X, axis=0), "sigma": np.std(X, axis=0)}
    elif norm_mode == "normal":
        return {"mode": "normal", "min": np.min(X, axis=0), "max": np.max(X, axis=0)}
    else:
        raise ValueError("Invalid normalization mode selected!")


def f_apply_Normalization(X, params):
    """Apply previously-fit normalization parameters (see f_get_norm_params)."""
    X = np.asarray(X, dtype=float).copy()
    num_Feature = X.shape[1]

    if params["mode"] == "standard":
        mu, sigma = params["mu"], params["sigma"]
        for j in range(num_Feature):
            if sigma[j] != 0:
                X[:, j] = (X[:, j] - mu[j]) / sigma[j]
            else:
                X[:, j] = X[:, j] - mu[j]
    elif params["mode"] == "normal":
        xmin, xmax = params["min"], params["max"]
        for j in range(num_Feature):
            denom = xmax[j] - xmin[j]
            X[:, j] = (X[:, j] - xmin[j]) / denom if denom != 0 else 0.0
    else:
        raise ValueError("Invalid normalization mode selected!")

    return X


def f_get_Normalization(X, norm_mode):
    """Fit-and-apply normalization on a single matrix (full-data convenience).

    Kept for backward compatibility. Prefer f_get_norm_params (fit on the
    training split) + f_apply_Normalization to avoid train/test leakage.
    """
    return f_apply_Normalization(X, f_get_norm_params(X, norm_mode))


### MASK FUNCTIONS ###
def f_get_fc_mask2(time, label, num_Event, num_Category):
    mask = np.zeros([time.shape[0], num_Event, num_Category])

    for i in range(time.shape[0]):
        if label[i, 0] != 0:  # If not censored
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:  # If censored
            mask[i, :, int(time[i, 0] + 1) :] = 1  # Fill 1 after censoring time

    # Debugging: Print some examples of the mask
    print(f"Mask 1 [0]:\n{mask[0]}")
    print(f"Mask 1 [1]:\n{mask[1]}")

    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    """
    mask5 is required to calculate the ranking loss (for pair-wise comparison)
    mask5 size is [N, num_Category].
    - For longitudinal measurements:
         1's from the last measurement to the event time (exclusive and inclusive, respectively)
    - For single measurement:
         1's from start to the event time (inclusive)
    """
    mask = np.zeros([np.shape(time)[0], num_Category])  # Initialize the mask

    # If longitudinal measurements exist
    if (
        isinstance(meas_time, np.ndarray) and np.shape(meas_time)[0] > 0
    ):  # Check if meas_time is an array
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])  # Last measurement time
            t2 = int(time[i, 0])  # Censoring/event time
            mask[i, (t1 + 1) : (t2 + 1)] = (
                1  # Excludes the last measurement time and includes the event time
            )

    else:  # Single measurement case
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])  # Censoring/event time
            mask[i, : (t + 1)] = 1  # Includes the event/censoring time

    return mask


### DATA IMPORT FUNCTIONS ###
def import_dataset_SYNTHETIC(norm_mode="standard"):
    """
    Load and preprocess the synthetic dataset.

    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)

    Returns: tuple (DIM, DATA, MASK)
    """
    in_filename = "./sample data/SYNTHETIC/synthetic_comprisk.csv"
    df = pd.read_csv(in_filename, sep=",")

    label = np.asarray(df[["label"]])
    time = np.asarray(df[["time"]])
    data = np.asarray(df.iloc[:, 4:])
    # NOTE: normalization is intentionally deferred to AFTER the train/test
    # split (fit on train only) in get_main.py / summarize_results.py to avoid
    # leakage. norm_mode is accepted for API compatibility but unused here.

    num_Category = int(np.max(time) * 1.2)  # To have enough time-horizon
    num_Event = int(
        len(np.unique(label)) - 1
    )  # Only count the number of events (do not count censoring)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = x_dim
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK


def import_dataset_METABRIC(norm_mode="standard"):
    """
    Load and preprocess the METABRIC dataset.

    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)

    Returns: tuple (DIM, DATA, MASK)
    """
    in_filename1 = "./sample data/METABRIC/cleaned_features_final.csv"
    in_filename2 = "./sample data/METABRIC/label.csv"

    df1 = pd.read_csv(in_filename1, sep=",")
    df2 = pd.read_csv(in_filename2, sep=",")

    data = np.asarray(df1)
    # NOTE: normalization is deferred to AFTER the train/test split (fit on
    # train only) to avoid leakage; see import_dataset_SYNTHETIC.

    time = np.asarray(df2[["event_time"]])
    # The reference repo ships this line commented out, but EVAL_TIMES=[144, 288, 432]
    # in main_RandomSearch.py only makes sense once event_time is rescaled to
    # "approximate months" (days/12). Without the conversion those horizons
    # capture <5% of events; with it they capture ~47/75/95%.
    time = np.round(time / 12.0)
    label = np.asarray(df2[["label"]])

    num_Category = int(np.max(time) * 1.2)  # To have enough time-horizon
    num_Event = int(
        len(np.unique(label)) - 1
    )  # Only count the number of events (do not count censoring)

    print(
        f"num_Category: {num_Category}, num_Event: {num_Event}, x_dim: {data.shape[1]}"
    )

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    # Debugging: Print mask shapes
    print(f"Mask 1 Shape: {mask1.shape}")
    print(f"Mask 2 Shape: {mask2.shape}")

    DIM = x_dim
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK
