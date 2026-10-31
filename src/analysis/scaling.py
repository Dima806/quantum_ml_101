"""The 2^N wall: statevector memory and runtime vs qubit count.

A statevector is 2^N complex128 numbers at 16 bytes each. Memory doubles with
every qubit: ~16 MB at 20 qubits, ~1 GB at 26, more than exists on Earth at 40.
You cannot simulate your way to scale.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from src.sim import gates
from src.sim.statevector import StateVector

_BYTES_PER_AMPLITUDE = 16  # complex128


def statevector_bytes(n_qubits: int) -> int:
    return (2**n_qubits) * _BYTES_PER_AMPLITUDE


def apply_time(n_qubits: int, repeats: int = 3) -> float:
    """Best-of-N wall-clock seconds to apply a Hadamard to every qubit."""
    best = float("inf")
    for _ in range(repeats):
        sv = StateVector(n_qubits)
        start = time.perf_counter()
        for q in range(n_qubits):
            sv.apply(gates.H, (q,))
        best = min(best, time.perf_counter() - start)
    return best


def scan(qubit_counts: Iterable[int]) -> dict[int, dict[str, float]]:
    return {
        n: {"bytes": float(statevector_bytes(n)), "seconds": apply_time(n)} for n in qubit_counts
    }
