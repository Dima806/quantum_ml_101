"""Classical baselines: logistic regression, RBF-kernel SVM, small MLP.

The honest question is never "does the quantum model work" but "does it beat the
classical one," so every arena needs these beside it.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def rbf_svm() -> SVC:
    return SVC(kernel="rbf", gamma="scale")


def small_mlp(seed: int = 0) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed)


def classical_baselines(seed: int = 0) -> dict[str, Any]:
    return {"logistic": logistic(), "rbf_svm": rbf_svm(), "mlp": small_mlp(seed)}
