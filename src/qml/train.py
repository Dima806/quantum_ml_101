"""Gradient-descent training loop for the VQC (the same recipe as a neural net)."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from src.qml.vqc import Circuit, square_loss


def train_vqc(
    circuit: Circuit,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    steps: int = 50,
    lr: float = 0.1,
) -> tuple[np.ndarray, list[float]]:
    """Minimize the square loss by gradient descent; return params + loss history."""
    opt = qml.GradientDescentOptimizer(lr)
    history: list[float] = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(lambda p: square_loss(circuit, p, X, y), params)
        history.append(float(loss))
    return params, history
