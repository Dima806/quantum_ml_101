"""Shared plotting helpers so every figure in the repo reads as one system."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_amplitudes(state: np.ndarray, ax: Any = None) -> Any:
    """Bar chart of measurement probabilities over the 2^N basis states."""
    if ax is None:
        _, ax = plt.subplots()
    probs = np.abs(state) ** 2
    n = int(np.log2(len(state)))
    labels = [format(i, f"0{n}b") for i in range(len(state))]
    ax.bar(labels, probs)
    ax.set_xlabel("basis state")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    return ax


def plot_scaling(qubit_counts: list[int], memory_bytes: list[float], ax: Any = None) -> Any:
    """Statevector memory vs qubit count on a log axis — the 2^N wall."""
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(qubit_counts, memory_bytes, marker="o")
    ax.set_xlabel("qubits")
    ax.set_ylabel("statevector bytes")
    ax.set_title("The 2^N memory wall")
    return ax


def plot_barren(qubit_counts: list[int], variances: list[float], ax: Any = None) -> Any:
    """Gradient variance vs qubit count on a log axis — the barren plateau."""
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(qubit_counts, variances, marker="o")
    ax.set_xlabel("qubits")
    ax.set_ylabel("Var(dC/dtheta)")
    ax.set_title("The barren plateau")
    return ax


def plot_arena(names: list[str], accuracies: list[float], ax: Any = None) -> Any:
    """Accuracy bar chart for the quantum-vs-classical arena."""
    if ax is None:
        _, ax = plt.subplots()
    ax.barh(names, accuracies)
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    return ax
