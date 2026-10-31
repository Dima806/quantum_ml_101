"""Gradient variance decreases with qubit count — the barren plateau."""

from __future__ import annotations

from src.analysis.barren import scan_qubits


def test_gradient_variance_decreases_with_qubits() -> None:
    variances = scan_qubits([2, 4, 6], n_layers=10, n_samples=120, seed=0)
    values = [variances[n] for n in (2, 4, 6)]
    assert all(v > 0.0 for v in values)
    # The plateau: variance at 6 qubits is far below variance at 2 qubits.
    assert values[-1] < values[0]
