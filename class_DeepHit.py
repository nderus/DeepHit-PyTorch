import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-08


class Model_DeepHit(nn.Module):
    """PyTorch port of DeepHit (Lee et al., AAAI 2018).

    Mirrors the original TF1 reference implementation at chl8856/DeepHit:
    - Shared FC subnet, then residual concat of the raw input with the shared
      output, then one cause-specific subnet per event.
    - Dropout after every hidden activation inside the subnets, and once more
      before the output layer; drop rate = 1 - keep_prob.
    - Joint softmax over the flattened (event, time) axis -- the output is a
      single distribution over all event-time pairs per subject.
    """

    def __init__(self, input_dims, network_settings):
        super().__init__()

        self.x_dim = input_dims["x_dim"]
        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]
        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]
        self.active_fn = network_settings["active_fn"]
        self.initial_W = network_settings["initial_W"]
        # Default 0.6 matches the upstream random-search config.
        self.keep_prob = network_settings.get("keep_prob", 0.6)
        self.dropout_p = 1.0 - self.keep_prob

        self.reg_W = 1e-4  # L2 on hidden weights
        self.reg_W_out = 1e-4  # L1 on the output weight

        self.shared_layers = self._build_subnet(
            self.x_dim, self.h_dim_shared, self.h_dim_shared, self.num_layers_shared
        )
        # Residual: cause-specific subnets see [raw x, shared output].
        cs_in_dim = self.x_dim + self.h_dim_shared
        self.cause_specific_layers = nn.ModuleList(
            [
                self._build_subnet(
                    cs_in_dim, self.h_dim_CS, self.h_dim_CS, self.num_layers_CS
                )
                for _ in range(self.num_Event)
            ]
        )
        self.output_layer = nn.Linear(
            self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category
        )

        self.initialize_weights()

    @staticmethod
    def _build_subnet(in_dim, h_dim, o_dim, num_layers):
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, o_dim))
        else:
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(in_dim, h_dim))
                elif i < num_layers - 1:
                    layers.append(nn.Linear(h_dim, h_dim))
                else:
                    layers.append(nn.Linear(h_dim, o_dim))
        return nn.ModuleList(layers)

    def initialize_weights(self):
        for layer in self.shared_layers:
            self.initial_W(layer.weight)
            nn.init.zeros_(layer.bias)
        for cs in self.cause_specific_layers:
            for layer in cs:
                self.initial_W(layer.weight)
                nn.init.zeros_(layer.bias)
        self.initial_W(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _forward_subnet(self, x, layers):
        # Original create_FCNet: dropout after every hidden activation, but
        # not after the subnet's final layer.
        n = len(layers)
        for i, layer in enumerate(layers):
            x = self.active_fn(layer(x))
            if n > 1 and i < n - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def forward(self, x):
        shared_out = self._forward_subnet(x, self.shared_layers)
        h = torch.cat([x, shared_out], dim=1)

        cs_outs = [self._forward_subnet(h, cs) for cs in self.cause_specific_layers]
        out = torch.stack(cs_outs, dim=1).view(x.size(0), -1)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.output_layer(out)
        # Joint softmax over the flat (event, time) axis.
        out = F.softmax(out, dim=1)
        return out.view(-1, self.num_Event, self.num_Category)

    def loss_log_likelihood(self, k_mb, m1_mb, predictions):
        # m1_mb encodes both branches: a single 1 at (k-1, t) for uncensored
        # subjects, and 1s at all (event, time > t_c) for censored subjects.
        # So sum_{event, time} m1 * pred = P(T=t, K=k) or P(T > t_c) as needed.
        masked_sum = torch.sum(m1_mb * predictions, dim=(1, 2))
        return -torch.mean(torch.log(masked_sum + _EPSILON))

    def loss_ranking(self, t_mb, k_mb, m2_mb, predictions):
        sigma1 = torch.tensor(0.1, dtype=torch.float32, device=predictions.device)
        eta = []

        for e in range(self.num_Event):
            one_vector = torch.ones_like(t_mb, dtype=torch.float32)

            I_2 = (k_mb == (e + 1)).float()
            I_2_diag = torch.diag(I_2.squeeze())

            tmp_e = predictions[:, e, :]

            R = torch.matmul(tmp_e, m2_mb.T)
            diag_R = torch.diag(R)
            R = torch.matmul(one_vector, diag_R.unsqueeze(0)) - R
            R = R.T

            T = F.relu(
                torch.sign(
                    torch.matmul(one_vector, t_mb.T) - torch.matmul(t_mb, one_vector.T)
                )
            )
            T = torch.matmul(I_2_diag, T)

            exp_term = torch.exp(-R / sigma1)
            tmp_eta = torch.mean(T * exp_term, dim=1, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)
        eta = torch.mean(eta.view(-1, self.num_Event), dim=1, keepdim=True)
        return torch.sum(eta)

    def loss_calibration(self, k_mb, m2_mb, predictions):
        eta_calibration = []
        for e in range(self.num_Event):
            I_2 = (k_mb == (e + 1)).float()
            tmp_e = predictions[:, e, :]
            r = torch.sum(tmp_e * m2_mb, dim=1)
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)
            eta_calibration.append(tmp_eta)
        eta_calibration = torch.stack(eta_calibration, dim=1)
        eta_calibration = torch.mean(
            eta_calibration.view(-1, self.num_Event), dim=1, keepdim=True
        )
        return torch.sum(eta_calibration)

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        loss1 = self.loss_log_likelihood(k_mb, m1_mb, predictions)
        loss2 = self.loss_ranking(t_mb, k_mb, m2_mb, predictions)
        loss3 = self.loss_calibration(k_mb, m2_mb, predictions)
        total_loss = alpha * loss1 + beta * loss2 + gamma * loss3

        # L2 over hidden weights only (biases excluded, matching upstream).
        l2_reg = torch.tensor(0.0, device=predictions.device)
        for layer in self.shared_layers:
            l2_reg = l2_reg + torch.sum(layer.weight**2)
        for cs in self.cause_specific_layers:
            for layer in cs:
                l2_reg = l2_reg + torch.sum(layer.weight**2)

        # L1 on the output weight only.
        l1_reg = torch.sum(torch.abs(self.output_layer.weight))

        return total_loss + self.reg_W * l2_reg + self.reg_W_out * l1_reg

    def training_step(self, DATA, MASK, PARAMETERS, optimizer):
        x_mb, _, _ = DATA
        optimizer.zero_grad()
        predictions = self(x_mb)
        loss = self.compute_loss(DATA, MASK, PARAMETERS, predictions)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            return self.forward(x_test)
