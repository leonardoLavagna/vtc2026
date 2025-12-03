import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from core.utils import expectation_z
from config.hyperparams import ORDER, V_MAX


# Trainable mapping
class TrainableMapping(nn.Module):
    def __init__(self, init_gamma=5.0, init_alpha=5.0, init_beta=0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))

    def forward(self, z):
        return self.gamma * torch.tanh(self.alpha * z + self.beta)


# Smoothing layer
class SmoothingFilter(nn.Module):
    def __init__(self, channels, kernel_size=21):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.channels = channels
        self.pad = kernel_size // 2
        # Depthwise convolution: one independent filter per channel
        self.filter = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,     
            padding=self.pad,
            bias=False
        )
        # Gaussian kernel initialization
        with torch.no_grad():
            t = torch.linspace(-2, 2, kernel_size)
            g = torch.exp(-t**2)
            g = g / g.sum()
            for c in range(channels):
                self.filter.weight[c, 0, :] = g

    def forward(self, y):
        # y shape: (T, channels)
        y = y.transpose(0, 1).unsqueeze(0)   
        y_f = self.filter(y)
        return y_f.squeeze(0).transpose(0, 1) 


# PQC ansatz
class TrainableAnsatz:
    def __init__(self, n_qubits, depth, gate_set):
        self.n_qubits = n_qubits
        self.depth = depth
        self.gate_set = gate_set
        self.params = {
            (l,q): {
                "ry": Parameter(f"ry_{l}_{q}") if "ry" in gate_set else None,
                "rz": Parameter(f"rz_{l}_{q}") if "rz" in gate_set else None,
            }
            for l in range(depth)
            for q in range(n_qubits)
        }

    def parameter_list(self):
        lst = []
        for l in range(self.depth):
            for q in range(self.n_qubits):
                p = self.params[(l,q)]
                if p["ry"] is not None: lst.append(p["ry"])
                if p["rz"] is not None: lst.append(p["rz"])
        return lst

    def build(self):
        qc = QuantumCircuit(self.n_qubits)
        for l in range(self.depth):
            for q in range(self.n_qubits):
                if "ry" in self.gate_set:
                    qc.ry(self.params[(l,q)]["ry"], q)
                if "rz" in self.gate_set:
                    qc.rz(self.params[(l,q)]["rz"], q)
        return qc


# Feature map
class FeatureMap(nn.Module):
    def __init__(self, order=ORDER):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(order + 1))

    def forward(self, t):
        # Ensure t is a torch.Tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        powers = torch.stack([t**k for k in range(len(self.coeffs))])
        return torch.sum(self.coeffs * powers)


# Full model
class QuantumDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mapping = TrainableMapping()
        self.smoothing = SmoothingFilter(channels=config["n_qubits"], kernel_size=21)
        self.feature_map = FeatureMap(order=config["feature_order"])
        self.ansatz = TrainableAnsatz(
            config["n_qubits"],
            config["depth"],
            config["gate_set"]
        )
        self.ansatz_params = nn.Parameter(
            torch.randn(len(self.ansatz.parameter_list()))
        )
        self.mode = "full" if config["n_qubits"] == 4 else "velocity"

    def forward(self, t):
        phi = self.feature_map(t)
        qc = self.ansatz.build()
        for q in range(self.config["n_qubits"]):
            qc.ry(float(phi), q)
        bind_vals = {
            p: float(self.ansatz_params[i])
            for i, p in enumerate(self.ansatz.parameter_list())
        }
        qc = qc.assign_parameters(bind_vals)
        expvals = expectation_z(qc, self.config["n_qubits"])
        return self.mapping(torch.tensor(expvals, dtype=torch.float32))

    def forward_sequence(self, times):
        outs = []
        for t in times:
            out_t = self.forward(float(t))     
            outs.append(out_t)
        Y = torch.stack(outs, dim=0)         
        Y_smooth = self.smoothing(Y)
        if self.mode == "velocity":
            # Only channel 0 is meaningful
            v_raw = Y_smooth[:, 0]            
            # Enforce physical range [0, V_MAX]
            v = V_MAX * torch.sigmoid(v_raw)
            return v.unsqueeze(1)            
        else:
            # Full physics: u, v, a, x = 4 channels
            # Velocity must be forced positive
            Y_phys = Y_smooth.clone()
            raw_v = Y_phys[:, 1]
            v = V_MAX * torch.sigmoid(raw_v)
            Y_phys[:, 1] = v
            return Y_phys