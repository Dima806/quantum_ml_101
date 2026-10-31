"""Harness: assemble the full quantum-vs-classical lineup and run the arena."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.classical.baselines import classical_baselines
from src.evaluation.arena import ArenaResult, run_arena
from src.qml.feature_maps import zz_feature_map
from src.qml.kernels import QuantumKernelClassifier
from src.qml.vqc import VQCClassifier


def all_models(n_qubits: int, seed: int = 0) -> dict[str, Any]:
    """Classical baselines plus the quantum-kernel SVM and the VQC."""
    models: dict[str, Any] = dict(classical_baselines(seed))
    models["quantum_kernel"] = QuantumKernelClassifier(
        feature_map=zz_feature_map, n_qubits=n_qubits
    )
    models["vqc"] = VQCClassifier(n_qubits=n_qubits, seed=seed)
    return models


def compare(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0
) -> list[ArenaResult]:
    n_qubits = X.shape[1]
    return run_arena(all_models(n_qubits, seed), X, y, test_size=test_size, seed=seed)
