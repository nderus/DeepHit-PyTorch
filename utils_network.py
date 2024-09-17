import torch
import torch.nn as nn
import torch.nn.functional as F

### CONSTRUCT MULTICELL FOR MULTI-LAYER RNNS
def create_rnn_cell(num_units, num_layers, keep_prob, RNN_type):
    '''
        GOAL         : Create a multi-cell (including a single cell) to construct a multi-layer RNN
        num_units    : Number of units in each layer
        num_layers   : Number of layers in MulticellRNN
        keep_prob    : Keep probability [0, 1] (if None, dropout is not employed)
        RNN_type     : Either 'LSTM' or 'GRU'
    '''
    cells = []

    for _ in range(num_layers):
        if RNN_type == 'GRU':
            cell = nn.GRUCell(num_units, num_units)
        elif RNN_type == 'LSTM':
            cell = nn.LSTMCell(num_units, num_units)

        cells.append(cell)

    # PyTorch RNN/GRU/LSTM use dropout in their built-in modules, no need to explicitly add it to cells
    multi_rnn = nn.RNNBase(input_size=num_units, hidden_size=num_units, num_layers=num_layers, dropout=1-keep_prob, batch_first=True)

    return multi_rnn

### EXTRACT STATE OUTPUT OF MULTICELL-RNNS
def create_concat_state(state, num_layers, RNN_type):
    '''
        GOAL	     : Concatenate the tuple-type tensor (state) into a single tensor
        state        : Input state is a tuple of MulticellRNN (i.e. output of MulticellRNN)
                       Consists of only hidden states h for GRU and hidden states c and h for LSTM
        num_layers   : Number of layers in MulticellRNN
        RNN_type     : Either 'LSTM' or 'GRU'
    '''
    for i in range(num_layers):
        if RNN_type == 'LSTM':
            tmp = state[i][1]  # i-th layer, h state for LSTM
        elif RNN_type == 'GRU':
            tmp = state[i]  # i-th layer, h state for GRU
        else:
            raise ValueError('ERROR: WRONG RNN CELL TYPE')

        if i == 0:
            rnn_state_out = tmp
        else:
            rnn_state_out = torch.cat([rnn_state_out, tmp], dim=1)
    
    return rnn_state_out

### FEEDFORWARD NETWORK
def create_FCNet(inputs, num_layers, h_dim, h_fn, o_dim, o_fn, w_init=None, keep_prob=1.0, w_reg=None):
    '''
        GOAL             : Create a fully connected network with different specifications 
        inputs (tensor)  : Input tensor
        num_layers       : Number of layers in FCNet
        h_dim  (int)     : Number of hidden units
        h_fn             : Activation function for hidden layers (default: nn.ReLU)
        o_dim  (int)     : Number of output units
        o_fn             : Activation function for output layers (default: None)
        w_init           : Initialization for weight matrix (default: Xavier (GlorotUniform))
        keep_prob        : Keep probability [0, 1] (if 1.0, dropout is not employed)
    '''

    # Default activation function (hidden: ReLU, output: None)
    if h_fn is None:
        h_fn = nn.ReLU()
    if o_fn is None:
        o_fn = None

    # Use GlorotUniform (Xavier initialization) by default if w_init is not specified
    if w_init is None:
        w_init = nn.init.xavier_uniform_

    layers = []

    # Build the fully connected network
    for layer in range(num_layers):
        if layer == 0:
            # First hidden layer
            layers.append(nn.Linear(inputs.size(1), h_dim))
            layers.append(h_fn)
            if keep_prob < 1.0:
                layers.append(nn.Dropout(1 - keep_prob))
        elif layer < num_layers - 1:
            # Intermediate hidden layers
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(h_fn)
            if keep_prob < 1.0:
                layers.append(nn.Dropout(1 - keep_prob))
        else:
            # Output layer
            layers.append(nn.Linear(h_dim, o_dim))
            if o_fn:
                layers.append(o_fn)

    # Stack the layers
    fc_net = nn.Sequential(*layers)
    # Apply Xavier initialization
    fc_net.apply(lambda m: w_init(m.weight) if isinstance(m, nn.Linear) else None)

    return fc_net