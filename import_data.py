import numpy as np
import pandas as pd
import torch

##### USER-DEFINED FUNCTIONS #####
def f_get_Normalization(X, norm_mode):
    """
    Normalize the input data matrix X according to the selected normalization mode.
    
    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)
    """
    num_Patient, num_Feature = X.shape

    if norm_mode == 'standard':  # Zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = X[:, j] - np.mean(X[:, j])
    elif norm_mode == 'normal':  # Min-max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        raise ValueError("Invalid normalization mode selected!")

    return X

### MASK FUNCTIONS ###
def f_get_fc_mask2(time, label, num_Event, num_Category):
    """
    Create the mask needed for log-likelihood loss (MASK2).

    time: N x 1 array of event times
    label: N x 1 array of event/censoring labels (0 = censoring, 1, 2, ... = events)
    num_Event: number of competing events
    num_Category: number of time intervals (categories)

    Returns: N x num_Event x num_Category mask array
    """
    mask = np.zeros([time.shape[0], num_Event, num_Category])
    for i in range(time.shape[0]):
        if label[i, 0] != 0:  # If not censored
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:  # If censored
            mask[i, :, int(time[i, 0] + 1):] = 1  # Fill 1 after censoring time

    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required to calculate the ranking loss (for pair-wise comparison)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
        - For single measurement:
             1's from start to the event time (inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category])  # Initialize the mask

    # If longitudinal measurements exist
    if isinstance(meas_time, np.ndarray) and np.shape(meas_time)[0] > 0:  # Check if meas_time is an array
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])  # Last measurement time
            t2 = int(time[i, 0])  # Censoring/event time
            mask[i, (t1+1):(t2+1)] = 1  # Excludes the last measurement time and includes the event time

    else:  # Single measurement case
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])  # Censoring/event time
            mask[i, :(t+1)] = 1  # Includes the event/censoring time
    
    return mask

### DATA IMPORT FUNCTIONS ###
def import_dataset_SYNTHETIC(norm_mode='standard'):
    """
    Load and preprocess the synthetic dataset.
    
    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)
    
    Returns: tuple (DIM, DATA, MASK)
    """
    in_filename = './sample data/SYNTHETIC/synthetic_comprisk.csv'
    df = pd.read_csv(in_filename, sep=',')

    label = np.asarray(df[['label']])
    time = np.asarray(df[['time']])
    data = np.asarray(df.iloc[:, 4:])
    data = f_get_Normalization(data, norm_mode)

    num_Category = int(np.max(time) * 1.2)  # To have enough time-horizon
    num_Event = int(len(np.unique(label)) - 1)  # Only count the number of events (do not count censoring)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = (x_dim)
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK


def import_dataset_METABRIC(norm_mode='standard'):
    """
    Load and preprocess the METABRIC dataset.

    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)

    Returns: tuple (DIM, DATA, MASK)
    """
    in_filename1 = './sample data/METABRIC/cleaned_features_final.csv'
    in_filename2 = './sample data/METABRIC/label.csv'

    df1 = pd.read_csv(in_filename1, sep=',')
    df2 = pd.read_csv(in_filename2, sep=',')

    data = np.asarray(df1)
    data = f_get_Normalization(data, norm_mode)

    time = np.asarray(df2[['event_time']])
    label = np.asarray(df2[['label']])

    num_Category = int(np.max(time) * 1.2)  # To have enough time-horizon
    num_Event = int(len(np.unique(label)) - 1)  # Only count the number of events (do not count censoring)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = (x_dim)
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK