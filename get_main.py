import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from termcolor import colored
from sklearn.metrics import brier_score_loss

# Import user-defined utilities
import utils_network as utils
from class_DeepHit import Model_DeepHit
from utils_eval import weighted_c_index, weighted_brier_score

_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS #####
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return x / (y + _EPSILON)

def f_get_minibatch(mb_size, x, label, time, mask1, mask2):
    idx = np.random.choice(np.arange(np.shape(x)[0]), mb_size, replace=False)
    
    x_mb = x[idx, :].astype(np.float32)
    k_mb = label[idx, :].astype(np.float32)  # censoring(0)/event(1,2,..) label
    t_mb = time[idx, :].astype(np.float32)
    m1_mb = mask1[idx, :, :].astype(np.float32)  # fc_mask
    m2_mb = mask2[idx, :].astype(np.float32)  # fc_mask
    return torch.tensor(x_mb), torch.tensor(k_mb), torch.tensor(t_mb), torch.tensor(m1_mb), torch.tensor(m2_mb)


def get_valid_performance(DATA, MASK, in_parser, out_itr, eval_time=None, MAX_VALUE=-99, OUT_ITERATION=5, seed=1234):
    ##### DATA & MASK
    (data, time, label) = DATA
    (mask1, mask2) = MASK

    x_dim = np.shape(data)[1]
    _, num_Event, num_Category = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]

    ACTIVATION_FN = {'relu': F.relu, 'elu': F.elu, 'tanh': torch.tanh}

    ##### HYPER-PARAMETERS
    mb_size = in_parser['mb_size']
    iteration = in_parser['iteration']
    keep_prob = in_parser['keep_prob']
    lr_train = in_parser['lr_train']

    alpha = in_parser['alpha']  # for log-likelihood loss
    beta = in_parser['beta']  # for ranking loss
    gamma = in_parser['gamma']  # for RNN-prediction loss
    parameter_name = 'a' + str('%02.0f' % (10 * alpha)) + 'b' + str('%02.0f' % (10 * beta)) + 'c' + str('%02.0f' % (10 * gamma))

    # Xavier initializer is GlorotUniform in TensorFlow 2.x
    initial_W = torch.nn.init.xavier_uniform_

    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims = {
        'x_dim': x_dim,
        'num_Event': num_Event,
        'num_Category': num_Category
    }

    # NETWORK HYPER-PARAMETERS
    network_settings = {
        'h_dim_shared': in_parser['h_dim_shared'],
        'num_layers_shared': in_parser['num_layers_shared'],
        'h_dim_CS': in_parser['h_dim_CS'],
        'num_layers_CS': in_parser['num_layers_CS'],
        'active_fn': ACTIVATION_FN[in_parser['active_fn']],
        'initial_W': initial_W
    }

    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)

    # Create directories if they don't exist
    os.makedirs(file_path_final + '/models/', exist_ok=True)

    print(file_path_final + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')')

    ##### CREATE DEEPHIT NETWORK
    model = Model_DeepHit(input_dims, network_settings)
    optimizer = optim.Adam(model.parameters(), lr=lr_train)
    ### TRAINING-TESTING SPLIT
    (tr_data, te_data, tr_time, te_time, tr_label, te_label, 
    tr_mask1, te_mask1, tr_mask2, te_mask2) = train_test_split(
        data, time, label, mask1, mask2, test_size=0.20, random_state=seed)

    (tr_data, va_data, tr_time, va_time, tr_label, va_label, 
    tr_mask1, va_mask1, tr_mask2, va_mask2) = train_test_split(
        tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed)

    # Convert va_data to a tensor
    va_data = torch.tensor(va_data, dtype=torch.float32)  # ensure the data is a PyTorch tensor

    max_valid = -99
    stop_flag = 0

    if eval_time is None:
        eval_time = [int(np.percentile(tr_time, 25)), int(np.percentile(tr_time, 50)), int(np.percentile(tr_time, 75))]

    ### MAIN TRAINING LOOP
    print("MAIN TRAINING ...")
    print("EVALUATION TIMES: " + str(eval_time))

    avg_loss = 0
    for itr in range(iteration):
        if stop_flag > 5:  # Early stopping condition
            break

        # Fetch minibatch
        x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb)
        PARAMETERS = (alpha, beta, gamma)

        # Train the model on the current batch
        loss_curr = model.training_step(DATA, MASK, PARAMETERS, optimizer)
        avg_loss += loss_curr / 1000

        if (itr + 1) % 1000 == 0:
            print('|| ITR: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' % (avg_loss)), 'yellow', attrs=['bold']))
            avg_loss = 0
            ### VALIDATION (based on average C-index of our interest) ###
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                pred = model(va_data)

            ### EVALUATION ###
            va_result1 = np.zeros([num_Event, len(eval_time)])

            for t, t_time in enumerate(eval_time):
                eval_horizon = int(t_time)

                if eval_horizon >= num_Category:
                    print('ERROR: evaluation horizon is out of range')
                    va_result1[:, t] = -1
                else:
                    risk = torch.sum(pred[:, :, :(eval_horizon + 1)], dim=2).numpy()  # risk score until eval_time
                    for k in range(num_Event):
                        va_result1[k, t] = weighted_c_index(
                            tr_time, (tr_label[:, 0] == k + 1).astype(int),
                            risk[:, k], va_time, (va_label[:, 0] == k + 1).astype(int), eval_horizon)

            tmp_valid = np.mean(va_result1)

            if tmp_valid > max_valid:
                stop_flag = 0
                max_valid = tmp_valid
                print(f'Updated... Average C-index = {tmp_valid:.4f}')

                # Save model weights when validation improves
                torch.save(model.state_dict(), os.path.join(file_path_final, 'models', f'model_itr_{out_itr}.pth'))
            else:
                stop_flag += 1

    return max_valid