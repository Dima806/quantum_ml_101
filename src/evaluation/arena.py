"""Quantum vs classical: fit each model, measure test accuracy and fit runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class ArenaResult:
    name: str
    accuracy: float
    fit_seconds: float


def evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ArenaResult:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    return ArenaResult(name=name, accuracy=accuracy, fit_seconds=fit_seconds)


def run_arena(
    models: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    seed: int = 0,
) -> list[ArenaResult]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return [
        evaluate(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()
    ]


def results_frame(results: list[ArenaResult]) -> pd.DataFrame:
    """Tidy table, sorted best-accuracy first — the arena scoreboard."""
    frame = pd.DataFrame(
        {
            "model": [r.name for r in results],
            "accuracy": [r.accuracy for r in results],
            "fit_seconds": [r.fit_seconds for r in results],
        }
    )
    return frame.sort_values("accuracy", ascending=False).reset_index(drop=True)
