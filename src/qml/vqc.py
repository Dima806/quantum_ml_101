"""Variational quantum classifier: a parameterized circuit trained like a net.

The ansatz is layers of RY rotations with a ring of CNOT entanglers. Gradients
come from PennyLane's parameter-shift rule (an exact gradient that plays the
role backprop plays for a neural net); ``finite_diff_gradient`` is the numerical
check that the two agree.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

Circuit = Callable[[np.ndarray, np.ndarray], float]


def make_circuit(n_qubits: int, n_layers: int, diff_method: str = "parameter-shift") -> Circuit:
    """Build a QNode: angle-encode x, then n_layers of RY + CNOT ring,
    measuring <Z> on qubit 0 (a real number in [-1, 1])."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params, x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i], wires=i)
            if n_qubits > 1:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return circuit


def init_params(n_qubits: int, n_layers: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits))
    return pnp.array(values, requires_grad=True)


def predictions(circuit: Circuit, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.array([float(circuit(params, x)) for x in X])


def square_loss(circuit: Circuit, params: np.ndarray, X: np.ndarray, y: np.ndarray):
    """Mean squared error between <Z> predictions and +/-1 labels."""
    preds = pnp.stack([circuit(params, x) for x in X])
    return ((preds - y) ** 2).mean()


def gradient(circuit: Circuit, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Parameter-shift gradient of the loss w.r.t. every circuit parameter."""
    grad_fn = qml.grad(lambda p: square_loss(circuit, p, X, y))
    return np.array(grad_fn(params))


def finite_diff_gradient(
    circuit: Circuit,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference gradient — the numerical check."""
    base = np.array(params, dtype=float)
    grad = np.zeros_like(base)
    for idx in np.ndindex(base.shape):
        plus = base.copy()
        minus = base.copy()
        plus[idx] += eps
        minus[idx] -= eps
        f_plus = float(square_loss(circuit, pnp.array(plus), X, y))
        f_minus = float(square_loss(circuit, pnp.array(minus), X, y))
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


class VQCClassifier:
    """sklearn-style variational quantum classifier (binary, labels 0/1)."""

    def __init__(
        self,
        n_layers: int = 2,
        n_qubits: int | None = None,
        steps: int = 30,
        lr: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.steps = steps
        self.lr = lr
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> VQCClassifier:
        self.n_qubits_ = self.n_qubits or X.shape[1]
        self.circuit_ = make_circuit(self.n_qubits_, self.n_layers)
        rng = np.random.default_rng(self.seed)
        params = init_params(self.n_qubits_, self.n_layers, rng)
        labels = np.where(np.asarray(y) <= 0, -1.0, 1.0)
        opt = qml.GradientDescentOptimizer(self.lr)
        for _ in range(self.steps):
            params = opt.step(lambda p: square_loss(self.circuit_, p, X, labels), params)
        self.params_ = params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = predictions(self.circuit_, self.params_, X)
        return (preds > 0.0).astype(int)
