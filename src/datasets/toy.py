"""Toy datasets: standard low-dimensional sets where classical baselines are
strong, plus a planted "quantum-friendly" set engineered to suit the ZZ feature
map (so the project can show the engineered exception honestly, not as a win).

All sets are tiny, seeded, and scaled into [0, pi] so angles are well-behaved.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_circles, make_moons
from sklearn.preprocessing import MinMaxScaler


def _scale(X: np.ndarray) -> np.ndarray:
    return MinMaxScaler(feature_range=(0.0, float(np.pi))).fit_transform(X)


def moons(
    n_samples: int = 200, noise: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return _scale(X), y


def circles(
    n_samples: int = 200, noise: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    return _scale(X), y


def iris_2d() -> tuple[np.ndarray, np.ndarray]:
    """First two features of iris, classes 0 and 1 (a 2-qubit problem)."""
    data = load_iris()
    X = np.asarray(data.data)[:, :2]
    y = np.asarray(data.target)
    mask = y < 2
    return _scale(X[mask]), y[mask]


def planted_quantum(
    n_samples: int = 200,
    n_features: int = 2,
    seed: int = 0,
    margin: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """A set engineered to suit the ZZ fidelity kernel (the Havlicek "ad hoc"
    construction).

    Each point is ZZ-encoded into a state |phi(x)>, and the label is the sign of
    <phi(x)|M|phi(x)> relative to a threshold, where M is a fixed random
    observable with +/-1 eigenvalues. Because that decision function is linear in
    the fidelity-kernel feature space (the space of density matrices), a
    quantum-kernel SVM with the same feature map can separate it by design, while
    RBF/logistic on the raw 2-D angles cannot. Points within ``margin`` of the
    threshold are rejected so the classes are cleanly separable.

    This is an engineered exception, not a general quantum advantage — flag it.
    """
    from src.qml.feature_maps import zz_feature_map
    from src.sim.statevector import run_ops

    rng = np.random.default_rng(seed)
    n_qubits = n_features
    dim = 2**n_qubits

    # A Haar-ish random unitary V, then M = V diag(+1.../-1...) V^dagger.
    gaussian = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(gaussian)
    v = q @ np.diag(np.exp(1j * np.angle(np.diag(r))))
    spectrum = np.ones(dim)
    spectrum[dim // 2 :] = -1.0
    observable = (v * spectrum) @ v.conj().T

    pool = rng.uniform(0.0, np.pi, size=(n_samples * 4, n_features))
    values = np.array(
        [
            float(np.real(s.conj() @ observable @ s))
            for s in (run_ops(n_qubits, zz_feature_map(x)).state for x in pool)
        ]
    )
    threshold = float(np.median(values))
    keep = np.abs(values - threshold) > margin
    X = pool[keep][:n_samples]
    y = (values[keep][:n_samples] > threshold).astype(int)
    return X, y


def get_dataset(name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        return moons(seed=seed)
    if name == "circles":
        return circles(seed=seed)
    if name == "iris":
        return iris_2d()
    if name == "planted":
        return planted_quantum(seed=seed)
    raise ValueError(f"unknown dataset: {name}")
