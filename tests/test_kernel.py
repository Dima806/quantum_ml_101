"""The quantum kernel is symmetric and PSD, and plugs into a scikit-learn SVM."""

from __future__ import annotations

import numpy as np

from src.datasets.toy import moons
from src.qml.feature_maps import angle_encoding, zz_feature_map
from src.qml.kernels import QuantumKernelClassifier, kernel_matrix


def test_kernel_symmetric_psd_unit_diagonal() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, np.pi, size=(8, 2))
    gram = kernel_matrix(X, X, zz_feature_map, n_qubits=2)
    np.testing.assert_allclose(gram, gram.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(gram)
    assert eigenvalues.min() > -1e-8  # positive semidefinite


def test_plugs_into_sklearn_svm() -> None:
    X, y = moons(n_samples=40, seed=0)
    clf = QuantumKernelClassifier(feature_map=angle_encoding, n_qubits=2).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)
    assert set(np.unique(preds)).issubset({0, 1})
    # A sensible kernel SVM should at least fit its own training data decently.
    assert (preds == y).mean() > 0.6
