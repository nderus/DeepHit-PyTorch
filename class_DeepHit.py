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
        
        # Build shared and cause-specific subnetworks
        self.shared_layers = self.build_shared_layers()
        self.cause_specific_layers = self.build_cause_specific_layers()
        self.output_layer = self.build_output_layer()

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
        #print("Input shape:", x.shape)
        for layer in self.shared_layers:
            x = self.active_fn(layer(x))
            #print("After shared layer:", x.shape)
        #print("Using activation function:", self.active_fn)
        # Forward pass through cause-specific layers
        outputs = []
        for event_layers in self.cause_specific_layers:
            h = x  # Output from shared layers
            for layer in event_layers:
                h = self.active_fn(layer(h))
                #print("After event layer:", h.shape)
            outputs.append(h)
        # Stack outputs for each event and reshape
        out = torch.stack(outputs, dim=1)  # This gives you a tensor of shape [64, num_Event, h_dim_CS]
        #print('DEBUG1 out', out.shape)
        out = out.view(out.size(0), -1)    # Flatten for the output layer: [64, num_Event * h_dim_CS]
        #print('DEBUG2 out', out.shape)

        # Apply dropout and output layer
        out = F.dropout(out, p=0.4, training=self.training)  # Dropout rate 0.4 (keep_prob = 0.6)
        out = self.output_layer(out)  # Output layer expects [64, num_Event * num_Category]
        #print('DEBUG3 out', out.shape)

        out = out.view(-1, self.num_Event, self.num_Category)
        #print('DEBUG4 out', out.shape)

        return F.softmax(out, dim=-1)

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        I_1 = torch.sign(k_mb)

        # Compute log-likelihood loss (Loss 1)
        tmp1 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1)

        tmp2 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2)

        LOSS_1 = -torch.mean(tmp1 + 1.0 * tmp2)

        # Return the computed loss (you can include ranking and calibration loss as needed)
        return alpha * LOSS_1  # Add the other losses like ranking and calibration here

    def training_step(self, DATA, MASK, PARAMETERS, optimizer):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

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