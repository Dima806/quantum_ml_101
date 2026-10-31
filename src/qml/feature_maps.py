"""Feature maps: encode classical data into a quantum state.

A feature map is the quantum analog of lifting data into a feature space. Each
map takes a 1-D sample and returns a circuit (list of ops) on ``len(x)`` qubits.
"""

from __future__ import annotations

import numpy as np

from src.sim.statevector import Op


def angle_encoding(x: np.ndarray, scale: float = 1.0) -> list[Op]:
    """Angle encoding: RY(scale * x_i) on qubit i. One qubit per feature."""
    return [("RY", (i,), (float(scale * xi),)) for i, xi in enumerate(x)]


def zz_feature_map(x: np.ndarray, reps: int = 1) -> list[Op]:
    """ZZ entangling map (Qiskit-style): H, single-qubit Z rotations, and
    entangling ZZ interactions on neighboring qubits. Captures feature products,
    so it suits data with parity/product structure."""
    n = len(x)
    ops: list[Op] = []
    for _ in range(reps):
        for i in range(n):
            ops.append(("H", (i,), ()))
        for i in range(n):
            ops.append(("RZ", (i,), (float(2.0 * x[i]),)))
        for i in range(n - 1):
            angle = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
            ops.append(("CNOT", (i, i + 1), ()))
            ops.append(("RZ", (i + 1,), (float(angle),)))
            ops.append(("CNOT", (i, i + 1), ()))
    return ops
