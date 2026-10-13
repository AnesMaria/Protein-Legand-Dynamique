import math

import pennylane as qml
import torch
from torch import nn


class QuantumLayer(nn.Module):
    """Analytic CPU simulator with differentiable CPU/CUDA transfers."""
    def __init__(self, n_qubits=4, n_layers=2, angle_encoding="RY"):
        super().__init__()
        if n_qubits < 2 or n_layers < 1 or angle_encoding not in ("RY", "RX/RY/RZ"):
            raise ValueError("Require >=2 qubits, >=1 layers, encoding RY or RX/RY/RZ")
        self.n_qubits = n_qubits
        device = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for wire in range(n_qubits):
                if angle_encoding == "RX/RY/RZ":
                    qml.RX(inputs[..., wire], wires=wire)
                    qml.RY(inputs[..., wire], wires=wire)
                    qml.RZ(inputs[..., wire], wires=wire)
                else:
                    qml.RY(inputs[..., wire], wires=wire)
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)},
                                        init_method=lambda w: nn.init.uniform_(w, -0.1, 0.1))

    def forward(self, x):
        assert x.ndim == 2 and x.shape[-1] == self.n_qubits
        # TorchLayer preserves autograd through both transfers; simulator is always CPU.
        self.circuit.to("cpu")
        output = self.circuit((math.pi * torch.tanh(x)).to("cpu"))
        return output.to(device=x.device, dtype=x.dtype)
