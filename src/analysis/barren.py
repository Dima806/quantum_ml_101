"""The barren plateau: gradient variance collapses exponentially with qubits.

Sample random parameters for a random deep circuit, measure the variance of one
cost-gradient component across the landscape, and watch it halve with each added
qubit. This is a proven property of the landscape, not a bug — it is the measured
reason you cannot simply scale a variational circuit up to force an advantage.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Iterable

import numpy as np
import pennylane as qml

# A barren-plateau circuit takes a single parameter array and returns <obs>.
ParamCircuit = Callable[[np.ndarray], float]


def _global_z(n_qubits: int):
    """Observable Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1} (a global cost)."""
    return functools.reduce(operator.matmul, (qml.PauliZ(i) for i in range(n_qubits)))


def _random_circuit(n_qubits: int, n_layers: int) -> ParamCircuit:
    """RY rotations + a CZ ladder each layer, measuring a global Z observable."""
    dev = qml.device("default.qubit", wires=n_qubits)
    obs = _global_z(n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        return qml.expval(obs)

    return circuit


def gradient_variance(n_qubits: int, n_layers: int, n_samples: int, seed: int = 0) -> float:
    """Variance of dC/dtheta for one parameter over random parameter samples.

    Uses the parameter-shift rule directly: shifting a single angle by +/- pi/2
    gives the exact derivative with two forward evaluations.
    """
    circuit = _random_circuit(n_qubits, n_layers)
    rng = np.random.default_rng(seed)
    shift = np.zeros((n_layers, n_qubits))
    shift[0, 0] = np.pi / 2.0
    grads = np.empty(n_samples)
    for s in range(n_samples):
        params = rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits))
        grads[s] = 0.5 * (float(circuit(params + shift)) - float(circuit(params - shift)))
    return float(np.var(grads))


def scan_qubits(
    qubit_counts: Iterable[int], n_layers: int, n_samples: int, seed: int = 0
) -> dict[int, float]:
    """Gradient variance at each qubit count — the plateau as a curve."""
    return {n: gradient_variance(n, n_layers, n_samples, seed) for n in qubit_counts}
