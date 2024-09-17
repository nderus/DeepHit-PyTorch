import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_EPSILON = 1e-08

# USER-DEFINED FUNCTIONS
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return x / (y + _EPSILON)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(Model_DeepHit, self).__init__()

        # INPUT DIMENSIONS
        self.x_dim = input_dims['x_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']

        # NETWORK HYPER-PARAMETERS
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']
        self.active_fn = network_settings['active_fn']
        self.initial_W = network_settings['initial_W']  # To track the initializer, not used explicitly
        
        # Build shared and cause-specific subnetworks
        self.shared_layers = self.build_shared_layers()
        self.cause_specific_layers = self.build_cause_specific_layers()
        self.output_layer = self.build_output_layer()

        # Apply weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Apply the specified initializer (e.g., Xavier Uniform) to all Linear layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                self.initial_W(layer.weight)  # Apply initializer to the weights

    def build_shared_layers(self):
        layers = []
        for i in range(self.num_layers_shared):
            input_dim = self.x_dim if i == 0 else self.h_dim_shared
            layers.append(nn.Linear(input_dim, self.h_dim_shared))
        return nn.ModuleList(layers)

    def build_cause_specific_layers(self):
        layers = []
        for _ in range(self.num_Event):
            event_layers = []
            # The first event layer should take the shared layer output as input (self.h_dim_shared)
            for i in range(self.num_layers_CS):
                input_dim = self.h_dim_shared if i == 0 else self.h_dim_CS  # Only the first layer uses h_dim_shared
                event_layers.append(nn.Linear(input_dim, self.h_dim_CS))
            layers.append(nn.ModuleList(event_layers))
        return nn.ModuleList(layers)

    def build_output_layer(self):
        return nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)

    def forward(self, x):
        # Forward pass through shared layers
        for layer in self.shared_layers:
            x = self.active_fn(layer(x))

        # Forward pass through cause-specific layers
        outputs = []
        for event_layers in self.cause_specific_layers:
            h = x  # Output from shared layers
            for layer in event_layers:
                h = self.active_fn(layer(h))
            outputs.append(h)

        # Stack outputs for each event and reshape
        out = torch.stack(outputs, dim=1)  # [batch_size, num_Event, h_dim_CS]
        out = out.view(out.size(0), -1)    # Flatten for the output layer: [batch_size, num_Event * h_dim_CS]
        
        # Apply dropout and output layer
        out = F.dropout(out, p=0.4, training=self.training)  # Dropout rate 0.4 (keep_prob = 0.6)
        out = self.output_layer(out)  # Output layer expects [batch_size, num_Event * num_Category]

        out = out.view(-1, self.num_Event, self.num_Category)
        return F.softmax(out, dim=-1)

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        I_1 = torch.sign(k_mb)

        ### Loss 1: Log-likelihood loss
        tmp1 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1)

        tmp2 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2)

        LOSS_1 = -torch.mean(tmp1 + 1.0 * tmp2)

        ### Loss 2: Ranking loss
        sigma1 = torch.tensor(0.1, dtype=torch.float32)
        eta = []

        for e in range(self.num_Event):
            # Indicator for subjects with event e+1
            I_2 = (k_mb == (e + 1)).float()  # Shape: [batch_size, 1]
            
            # Diagonal matrix of the indicator (to filter T matrix)
            I_2_diag = torch.diag(I_2.squeeze())  # Shape: [batch_size, batch_size]

            # Event-specific predictions for event e
            tmp_e = predictions[:, e, :]  # Shape: [batch_size, num_Category]

            # Compute risk score differences (R_ij = r_i(T_i) - r_j(T_i))
            R = torch.matmul(tmp_e, m2_mb.T)  # Shape: [batch_size, batch_size] (matrix of risk differences)
            R = torch.clamp(R, min=-50, max=50)  # Ensures R values are not too large or small

            diag_R = torch.diag(R)  # Extract diagonal, Shape: [batch_size]
            R = diag_R.unsqueeze(1) - R  # R_ij = r_i(T_i) - r_j(T_i), Shape: [batch_size, batch_size]

            # Compare event times (T_ij = 1 if t_i < t_j)
            T = ((t_mb.unsqueeze(1) - t_mb) > 0).float()  # Shape: [batch_size, batch_size], 1 if t_i < t_j
            
            # Apply event indicator to only consider relevant subjects
            T = torch.matmul(I_2_diag, T)  # Shape: [batch_size, batch_size]

            # Ranking penalty (T * exp(-R / sigma1))
            exp_term = torch.exp(-R / sigma1)  # Safe to calculate exp after clamping R
            exp_term = torch.clamp(exp_term, min=1e-8, max=1e8)  # Further clamp exp values to avoid NaNs

            tmp_eta = torch.mean(T * exp_term, dim=1, keepdim=True)  # Shape: [batch_size, 1]
            eta.append(tmp_eta)

        # Stack and compute mean over subjects and events
        eta = torch.stack(eta, dim=1)  # Shape: [batch_size, num_Event, 1]
        eta = torch.mean(eta.view(-1, self.num_Event), dim=1, keepdim=True)  # Shape: [batch_size, 1]

        LOSS_2 = torch.sum(eta)

        ### Loss 3: Calibration loss
        eta_calibration = []

        for e in range(self.num_Event):
            I_2 = (k_mb == (e + 1)).float()
            tmp_e = predictions[:, e, :]  # Event-specific joint probability

            r = torch.sum(tmp_e * m2_mb, dim=1)  # Sum predicted probabilities up to the event time
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)  # MSE for this event

            eta_calibration.append(tmp_eta)

        eta_calibration = torch.stack(eta_calibration, dim=1)
        eta_calibration = torch.mean(eta_calibration.view(-1, self.num_Event), dim=1, keepdim=True)

        LOSS_3 = torch.sum(eta_calibration)

        ### Total Loss
        LOSS_TOTAL = alpha * LOSS_1 + beta * LOSS_2 + gamma * LOSS_3
        ### FAMO losses

        return LOSS_TOTAL
    
    def training_step(self, DATA, MASK, PARAMETERS, optimizer):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        #alpha, beta, gamma = PARAMETERS

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = self(x_mb)

        # Compute loss
        loss = self.compute_loss(DATA, MASK, PARAMETERS, predictions)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, x_test):
        self.eval()  # Set the model to evaluation mode (disables dropout, etc.)
        with torch.no_grad():  # Disable gradient computation during inference
            return self.forward(x_test)


from itertools import chain
from typing import Iterator

class Model_DeepHit_FAMO(Model_DeepHit):
    def __init__(self, input_dims, network_settings):
        super(Model_DeepHit_FAMO, self).__init__(input_dims, network_settings)

    # Override the training_step method to use FAMO optimization
    def training_step(self, DATA, MASK, PARAMETERS, optimizer):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        # Zero gradients (FAMO-specific)
        optimizer.zero_grad()

        # Forward pass
        predictions = self(x_mb)

        # Compute loss
        loss, losses = self.compute_loss(DATA, MASK, PARAMETERS, predictions)

        # Backward pass and optimization using FAMO optimizer
        loss.backward()
        optimizer.step()  # This step could include additional logic specific to FAMO

        return loss.item()
    

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        I_1 = torch.sign(k_mb)

        ### Loss 1: Log-likelihood loss
        tmp1 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1)

        tmp2 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2)

        LOSS_1 = -torch.mean(tmp1 + 1.0 * tmp2)

        ### Loss 2: Ranking loss
        sigma1 = torch.tensor(0.1, dtype=torch.float32)
        eta = []

        for e in range(self.num_Event):
            # Indicator for subjects with event e+1
            I_2 = (k_mb == (e + 1)).float()  # Shape: [batch_size, 1]
            
            # Diagonal matrix of the indicator (to filter T matrix)
            I_2_diag = torch.diag(I_2.squeeze())  # Shape: [batch_size, batch_size]

            # Event-specific predictions for event e
            tmp_e = predictions[:, e, :]  # Shape: [batch_size, num_Category]

            # Compute risk score differences (R_ij = r_i(T_i) - r_j(T_i))
            R = torch.matmul(tmp_e, m2_mb.T)  # Shape: [batch_size, batch_size] (matrix of risk differences)
            R = torch.clamp(R, min=-50, max=50)  # Ensures R values are not too large or small

            diag_R = torch.diag(R)  # Extract diagonal, Shape: [batch_size]
            R = diag_R.unsqueeze(1) - R  # R_ij = r_i(T_i) - r_j(T_i), Shape: [batch_size, batch_size]

            # Compare event times (T_ij = 1 if t_i < t_j)
            T = ((t_mb.unsqueeze(1) - t_mb) > 0).float()  # Shape: [batch_size, batch_size], 1 if t_i < t_j
            
            # Apply event indicator to only consider relevant subjects
            T = torch.matmul(I_2_diag, T)  # Shape: [batch_size, batch_size]

            # Ranking penalty (T * exp(-R / sigma1))
            exp_term = torch.exp(-R / sigma1)  # Safe to calculate exp after clamping R
            exp_term = torch.clamp(exp_term, min=1e-8, max=1e8)  # Further clamp exp values to avoid NaNs

            tmp_eta = torch.mean(T * exp_term, dim=1, keepdim=True)  # Shape: [batch_size, 1]
            eta.append(tmp_eta)

        # Stack and compute mean over subjects and events
        eta = torch.stack(eta, dim=1)  # Shape: [batch_size, num_Event, 1]
        eta = torch.mean(eta.view(-1, self.num_Event), dim=1, keepdim=True)  # Shape: [batch_size, 1]

        LOSS_2 = torch.sum(eta)

        ### Loss 3: Calibration loss
        eta_calibration = []

        for e in range(self.num_Event):
            I_2 = (k_mb == (e + 1)).float()
            tmp_e = predictions[:, e, :]  # Event-specific joint probability

            r = torch.sum(tmp_e * m2_mb, dim=1)  # Sum predicted probabilities up to the event time
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)  # MSE for this event

            eta_calibration.append(tmp_eta)

        eta_calibration = torch.stack(eta_calibration, dim=1)
        eta_calibration = torch.mean(eta_calibration.view(-1, self.num_Event), dim=1, keepdim=True)

        LOSS_3 = torch.sum(eta_calibration)

        ### Total Loss
        LOSS_TOTAL = alpha * LOSS_1 + beta * LOSS_2 + gamma * LOSS_3
        ### FAMO losses
        losses = torch.stack([LOSS_1, LOSS_2, LOSS_3], dim=0)
        return LOSS_TOTAL, losses

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        # Returns the parameters of all shared layers
        return chain(self.shared_layers.parameters())
    

    def cause_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        # Returns the parameters of all shared layers
        return chain(self.cause_specific_layers.parameters())
    
    def output_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        # Returns the parameters of all shared layers
        return chain(self.output_layer.parameters())
    
