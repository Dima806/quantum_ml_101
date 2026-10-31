"""Quantum kernels: a kernel is a kernel, and it plugs into a classical SVM.

The quantum kernel is the fidelity between two encoded states,
K(x, x') = |<phi(x)|phi(x')>|^2. It is symmetric and positive semidefinite (a
Gram matrix of pure-state overlaps), so it drops straight into
``sklearn.svm.SVC(kernel="precomputed")`` — the kernel trick unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.svm import SVC

from src.qml.feature_maps import angle_encoding
from src.sim.statevector import Op, run_ops

FeatureMap = Callable[[np.ndarray], list[Op]]


def encoded_state(x: np.ndarray, feature_map: FeatureMap, n_qubits: int) -> np.ndarray:
    return run_ops(n_qubits, feature_map(x)).state


def kernel_matrix(
    x1: np.ndarray, x2: np.ndarray, feature_map: FeatureMap, n_qubits: int
) -> np.ndarray:
    """Gram matrix of state fidelities between every pair of rows."""
    states1 = [encoded_state(x, feature_map, n_qubits) for x in x1]
    states2 = [encoded_state(x, feature_map, n_qubits) for x in x2]
    out = np.empty((len(states1), len(states2)), dtype=float)
    for i, a in enumerate(states1):
        for j, b in enumerate(states2):
            out[i, j] = np.abs(np.vdot(a, b)) ** 2
    return out


class QuantumKernelClassifier:
    """sklearn-style classifier: a precomputed quantum-kernel SVM."""

    def __init__(
        self,
        feature_map: FeatureMap = angle_encoding,
        n_qubits: int | None = None,
        C: float = 1.0,
    ) -> None:
        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self.C = C

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuantumKernelClassifier:
        self.n_qubits_ = self.n_qubits or x.shape[1]
        self.x_train_ = np.asarray(x)
        gram = kernel_matrix(self.x_train_, self.x_train_, self.feature_map, self.n_qubits_)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(gram, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        gram = kernel_matrix(np.asarray(x), self.x_train_, self.feature_map, self.n_qubits_)
        return self.svc_.predict(gram)
