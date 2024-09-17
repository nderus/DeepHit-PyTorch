import numpy as np
import pandas as pd
import torch
import random
import os
from termcolor import colored
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils
from class_DeepHit import Model_DeepHit
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score

_EPSILON = 1e-08


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                float(input)
                return True
            except ValueError:
                return False

        for line in f.readlines():
            if ':' in line:
                key, value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass  # Deal with bad lines of text here    
    return data


##### MAIN SETTING #####
OUT_ITERATION = 5
data_mode = 'SYNTHETIC'  # METABRIC, SYNTHETIC
seed = 1234
EVAL_TIMES = [12, 24, 36]  # Evaluation times (for C-index and Brier-Score)

##### IMPORT DATASET #####
if data_mode == 'SYNTHETIC':
    x_dim, DATA, MASK = impt.import_dataset_SYNTHETIC(norm_mode='standard')
    EVAL_TIMES = [12, 24, 36]
elif data_mode == 'METABRIC':
    x_dim, DATA, MASK = impt.import_dataset_METABRIC(norm_mode='standard')
    EVAL_TIMES = [144, 288, 432]
else:
    raise ValueError('ERROR: DATA_MODE NOT FOUND !!!')

data, time, label = DATA
mask1, mask2 = MASK
_, num_Event, num_Category = mask1.shape  # Dimension of mask1: [subj, Num_Event, Num_Category]

in_path = os.path.join(data_mode, 'results/')
if not os.path.exists(in_path):
    os.makedirs(in_path)

FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL2 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])

for out_itr in range(OUT_ITERATION):
    in_hypfile = os.path.join(in_path, f'itr_{out_itr}/hyperparameters_log.txt')
    in_parser = load_logging(in_hypfile)

    ##### HYPER-PARAMETERS #####
    mb_size = in_parser['mb_size']
    iteration = in_parser['iteration']
    keep_prob = in_parser['keep_prob']
    lr_train = in_parser['lr_train']

    h_dim_shared = in_parser['h_dim_shared']
    h_dim_CS = in_parser['h_dim_CS']
    num_layers_shared = in_parser['num_layers_shared']
    num_layers_CS = in_parser['num_layers_CS']

    active_fn_dict = {'relu': torch.nn.ReLU(), 'elu': torch.nn.ELU(), 'tanh': torch.nn.Tanh()}
    if in_parser['active_fn'] in active_fn_dict:
        active_fn = active_fn_dict[in_parser['active_fn']]
    else:
        raise ValueError('Invalid activation function!')

    # Initialize the weights (equivalent to GlorotNormal in TensorFlow)
    initial_W = torch.nn.init.xavier_normal_

    alpha = in_parser['alpha']  # for log-likelihood loss
    beta = in_parser['beta']  # for ranking loss
    gamma = in_parser['gamma']  # for RNN-prediction loss
    parameter_name = f'a{10 * alpha:02.0f}b{10 * beta:02.0f}c{10 * gamma:02.0f}'

    ##### MAKE DICTIONARIES #####
    input_dims = {
        'x_dim': x_dim,
        'num_Event': num_Event,
        'num_Category': num_Category
    }

    network_settings = {
        'h_dim_shared': h_dim_shared,
        'h_dim_CS': h_dim_CS,
        'num_layers_shared': num_layers_shared,
        'num_layers_CS': num_layers_CS,
        'active_fn': active_fn,
        'initial_W': initial_W
    }

    ##### CREATE DEEPHIT NETWORK #####
    model = Model_DeepHit("DeepHit", input_dims, network_settings)

    ### TRAINING-TESTING SPLIT ###
    (tr_data, te_data, tr_time, te_time, tr_label, te_label,
     tr_mask1, te_mask1, tr_mask2, te_mask2) = train_test_split(
        data, time, label, mask1, mask2, test_size=0.20, random_state=seed)

    (tr_data, va_data, tr_time, va_time, tr_label, va_label,
     tr_mask1, va_mask1, tr_mask2, va_mask2) = train_test_split(
        tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed)

    ##### LOAD SAVED MODEL #####
    model.load_state_dict(torch.load(os.path.join(in_path, f'itr_{out_itr}/models/model_itr_{out_itr}.pt')))

    ### PREDICTION ###
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(te_data).float())

    ### EVALUATION ###
    result1, result2 = np.zeros([num_Event, len(EVAL_TIMES)]), np.zeros([num_Event, len(EVAL_TIMES)])

    for t, t_time in enumerate(EVAL_TIMES):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print('ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            risk = np.sum(pred[:, :, :(eval_horizon + 1)].cpu().numpy(), axis=2)  # Risk score until EVAL_TIMES
            for k in range(num_Event):
                result1[k, t] = weighted_c_index(tr_time, (tr_label[:, 0] == k + 1).astype(int),
                                                 risk[:, k], te_time, (te_label[:, 0] == k + 1).astype(int), eval_horizon)
                result2[k, t] = weighted_brier_score(tr_time, (tr_label[:, 0] == k + 1).astype(int),
                                                     risk[:, k], te_time, (te_label[:, 0] == k + 1).astype(int), eval_horizon)

    FINAL1[:, :, out_itr] = result1
    FINAL2[:, :, out_itr] = result2

    ### SAVE RESULTS ###
    row_header = [f'Event_{t + 1}' for t in range(num_Event)]
    col_header1 = [f'{t}yr c_index' for t in EVAL_TIMES]
    col_header2 = [f'{t}yr B_score' for t in EVAL_TIMES]

    df1 = pd.DataFrame(result1, index=row_header, columns=col_header1)
    df1.to_csv(os.path.join(in_path, f'result_CINDEX_itr{out_itr}.csv'))

    df2 = pd.DataFrame(result2, index=row_header, columns=col_header2)
    df2.to_csv(os.path.join(in_path, f'result_BRIER_itr{out_itr}.csv'))

    ### PRINT RESULTS ###
    print('========================================================')
    print(f'ITR: {out_itr + 1} DATA MODE: {data_mode} (a:{alpha} b:{beta} c:{gamma})')
    print(f'SharedNet Parameters: h_dim_shared = {h_dim_shared}, num_layers_shared = {num_layers_shared}, Non-Linearity: {active_fn}')
    print(f'CSNet Parameters: h_dim_CS = {h_dim_CS}, num_layers_CS = {num_layers_CS}, Non-Linearity: {active_fn}')
    print('--------------------------------------------------------')
    print('- C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')
    print('- BRIER-SCORE: ')
    print(df2)
    print('========================================================')


### FINAL MEAN/STD ###
df1_mean = pd.DataFrame(np.mean(FINAL1, axis=2), index=row_header, columns=col_header1)
df1_std = pd.DataFrame(np.std(FINAL1, axis=2), index=row_header, columns=col_header1)
df1_mean.to_csv(os.path.join(in_path, 'result_CINDEX_FINAL_MEAN.csv'))
df1_std.to_csv(os.path.join(in_path, 'result_CINDEX_FINAL_STD.csv'))

df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index=row_header, columns=col_header2)
df2_std = pd.DataFrame(np.std(FINAL2, axis=2), index=row_header, columns=col_header2)
df2_mean.to_csv(os.path.join(in_path, 'result_BRIER_FINAL_MEAN.csv'))
df2_std.to_csv(os.path.join(in_path, 'result_BRIER_FINAL_STD.csv'))

### PRINT FINAL RESULTS ###
print('========================================================')
print('- FINAL C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL BRIER-SCORE: ')
print(df2_mean)
print('========================================================')