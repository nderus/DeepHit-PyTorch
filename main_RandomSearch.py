import os
import random
import numpy as np
import torch
import get_main  # PyTorch implementation of get_main
import import_data as impt  # PyTorch-compatible data import functions

##### SET SEED FOR REPRODUCIBILITY #####
def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

##### SAVE AND LOAD LOGGING #####
# This saves the current hyperparameters
def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            f.write(f'{key}:{value}\n')

# This opens and calls the saved hyperparameters
def load_logging(filename):
    data = {}
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
                pass  # Handle bad lines of text here
    return data

##### RANDOM HYPERPARAMETER SELECTION #####
def get_random_hyperparameters(out_path):
    SET_BATCH_SIZE    = [32, 64, 128]  # Batch sizes
    SET_LAYERS        = [1, 2, 3, 5]  # Number of layers
    SET_NODES         = [50, 100, 200, 300]  # Number of nodes per layer
    SET_ACTIVATION_FN = ['relu', 'elu', 'tanh']  # Activation functions
    SET_ALPHA         = [0.1, 0.5, 1.0, 3.0, 5.0]  # Alpha for log-likelihood loss
    SET_BETA          = [0.1, 0.5, 1.0, 3.0, 5.0]  # Beta for ranking loss
    SET_GAMMA         = [0.1, 0.5, 1.0, 3.0, 5.0]  # Gamma for calibration loss

    new_parser = {
        'mb_size': random.choice(SET_BATCH_SIZE),
        'iteration': 50000,
        'keep_prob': 0.6,
        'lr_train': 1e-4,
        'h_dim_shared': random.choice(SET_NODES),
        'h_dim_CS': random.choice(SET_NODES),
        'num_layers_shared': random.choice(SET_LAYERS),
        'num_layers_CS': random.choice(SET_LAYERS),
        'active_fn': random.choice(SET_ACTIVATION_FN),
        'alpha': 1.0,  # Default alpha (change beta and gamma)
        'beta': random.choice(SET_BETA),
        'gamma': 0,  # Default (no calibration loss)
        'out_path': out_path
    }
    
    return new_parser  # Outputs the dictionary of the randomly-chosen hyperparameters

##### MAIN RANDOM SEARCH SETUP #####
OUT_ITERATION = 5  # Number of outer iterations (splits)
RS_ITERATION = 50  # Number of random search iterations

data_mode = 'METABRIC'
seed = 1234

##### IMPORT DATASET #####
'''
DATA FORMAT:
    num_Category  -> Time horizon (typically max event/censoring time * 1.2)
    num_Event     -> Number of events (len(np.unique(label)) - 1)
    x_dim         -> Feature dimensions
    EVAL_TIMES    -> List of evaluation times for validation
'''

if data_mode == 'SYNTHETIC':
    x_dim, DATA, MASK = impt.import_dataset_SYNTHETIC(norm_mode='standard')
    EVAL_TIMES = [12, 24, 36]  # Example evaluation times
elif data_mode == 'METABRIC':
    x_dim, DATA, MASK = impt.import_dataset_METABRIC(norm_mode='standard')
    EVAL_TIMES = [144, 288, 432]  # Example evaluation times
else:
    raise ValueError('ERROR: DATA_MODE NOT FOUND !!!')

# Unpack the dataset
data, time, label = DATA
mask1, mask2 = MASK

# Set output path for saving results
out_path = os.path.join(data_mode, 'results')

##### RANDOM SEARCH ACROSS MULTIPLE OUTER ITERATIONS #####
for itr in range(OUT_ITERATION):
    set_seeds(itr)  # Ensure reproducibility by setting seed at each outer iteration
    itr_dir = os.path.join(out_path, f'itr_{itr}')
    
    if not os.path.exists(itr_dir):
        os.makedirs(itr_dir)
    
    max_valid = 0.0
    log_name = os.path.join(itr_dir, 'hyperparameters_log.txt')

    for r_itr in range(RS_ITERATION):
        print(f'OUTER_ITERATION: {itr}')
        print(f'Random search... iteration: {r_itr}')

        # Randomly choose hyperparameters
        new_parser = get_random_hyperparameters(out_path)
        print(new_parser)

        # Get validation performance given the hyperparameters
        tmp_max = get_main.get_valid_performance(DATA, MASK, new_parser, itr, EVAL_TIMES, MAX_VALUE=max_valid)

        # If new max validation score is found, log the best hyperparameters
        if tmp_max > max_valid:
            max_valid = tmp_max
            max_parser = new_parser
            save_logging(max_parser, log_name)  # Save the hyperparameters if validation performance improves

        print(f'Current best: {max_valid}')