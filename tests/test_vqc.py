"""The parameter-shift gradient matches finite differences."""

from __future__ import annotations

import numpy as np

from src.qml.vqc import (
    VQCClassifier,
    finite_diff_gradient,
    gradient,
    init_params,
    make_circuit,
)


def test_parameter_shift_matches_finite_difference() -> None:
    rng = np.random.default_rng(0)
    n_qubits, n_layers = 2, 2
    circuit = make_circuit(n_qubits, n_layers)
    params = init_params(n_qubits, n_layers, rng)
    X = rng.uniform(0.0, np.pi, size=(5, n_qubits))
    y = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])

    g_shift = gradient(circuit, params, X, y)
    g_fd = finite_diff_gradient(circuit, params, X, y)

    np.testing.assert_allclose(g_shift, g_fd, atol=1e-5, rtol=1e-3)


def test_vqc_classifier_runs() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, np.pi, size=(20, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    clf = VQCClassifier(n_qubits=2, n_layers=2, steps=10, seed=0).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
