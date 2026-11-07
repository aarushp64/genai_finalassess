"""
NeuroTutor AI - PyTorch Implementation & Interactive Demonstration
Author: Aarush Parate (Roll No: 1)
Unit IV: Generative AI Tools and Techniques
A transparent, interactive educational PyTorch neural network demonstrating forward pass, loss computation, and gradient backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List


class EducationalMLP(nn.Module):
    """
    Educational 3-layer Feedforward Neural Network in PyTorch.
    Demonstrates layer transformations, activations, and parameter tracking.
    """

    def __init__(self, in_features: int = 4, hidden_1: int = 8, hidden_2: int = 6, out_features: int = 2):
        super(EducationalMLP, self).__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2, out_features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Executes forward pass while caching intermediate activation tensors for inspection.
        """
        z1 = self.fc1(x)
        a1 = self.relu1(z1)
        z2 = self.fc2(a1)
        a2 = self.relu2(z2)
        z3 = self.fc3(a2)
        probs = torch.softmax(z3, dim=-1)

        return {
            "input": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "logits": z3,
            "probabilities": probs,
        }


class PyTorchDemoEngine:
    """Manages interactive forward-backward computational graph walkthroughs."""

    def __init__(self):
        self.model = EducationalMLP()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)

    def run_step_demo(
        self, input_features: List[float], target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Performs a full educational cycle:
        1. Forward Pass
        2. Loss Calculation
        3. Backward Pass (Gradients)
        4. Optimization Step
        """
        # Prepare tensors
        x = torch.tensor([input_features], dtype=torch.float32)
        y = torch.tensor([target_class], dtype=torch.long)

        # Zero gradients
        self.optimizer.zero_grad()

        # 1. Forward Pass
        fwd = self.model(x)
        logits = fwd["logits"]
        probs = fwd["probabilities"]

        # 2. Loss Computation
        loss = self.criterion(logits, y)
        loss_val = float(loss.item())

        # 3. Backward Pass
        loss.backward()

        # Inspect Gradients
        grad_norms = {}
        weight_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = float(param.grad.norm(2).item())
            weight_norms[name] = float(param.norm(2).item())

        # 4. Parameter Update
        self.optimizer.step()

        return {
            "input_tensor": x.numpy().tolist()[0],
            "target_class": target_class,
            "predicted_class": int(torch.argmax(probs).item()),
            "class_probabilities": probs.detach().numpy().tolist()[0],
            "loss_value": round(loss_val, 5),
            "activation_shapes": {
                "Input (X)": list(x.shape),
                "Layer 1 Hidden (A1)": list(fwd["a1"].shape),
                "Layer 2 Hidden (A2)": list(fwd["a2"].shape),
                "Output Logits (Z3)": list(logits.shape),
            },
            "layer_1_activations": fwd["a1"].detach().numpy().tolist()[0],
            "layer_2_activations": fwd["a2"].detach().numpy().tolist()[0],
            "gradient_norms": grad_norms,
            "weight_norms": weight_norms,
        }

    @staticmethod
    def create_gradient_flow_chart(gradient_norms: Dict[str, float]) -> go.Figure:
        """Visualizes gradient magnitudes across model layers to demonstrate backpropagation."""
        layer_names = list(gradient_norms.keys())
        norms = list(gradient_norms.values())

        fig = go.Figure(
            data=[
                go.Bar(
                    x=layer_names,
                    y=norms,
                    marker=dict(
                        color="#8BAFD0",
                        line=dict(color="#A4C4E2", width=1.5),
                    ),
                    hovertemplate="Layer: %{x}<br>Gradient Norm L2: %{y:.5f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="PyTorch Backpropagation: L2 Gradient Norms (dL/dW)",
                font=dict(color="#F2F4F7", size=14),
            ),
            xaxis=dict(title="Parameter Tensor", color="#AAB5C4", tickangle=-20),
            yaxis=dict(title="Gradient Magnitude (L2 Norm)", color="#AAB5C4", gridcolor="#39465A"),
            paper_bgcolor="#171D29",
            plot_bgcolor="#202838",
            margin=dict(l=50, r=30, t=50, b=60),
            height=320,
        )
        return fig
