"""Single- and two-qubit gate unitaries as numpy complex128 matrices.

Conventions match PennyLane exactly so the from-scratch simulator can be
validated against it to machine precision:

    RX(t) = exp(-i t X / 2),  RY(t) = exp(-i t Y / 2),  RZ(t) = exp(-i t Z / 2).

Parameter-free gates are module constants; parameterized gates are functions.
"""

from __future__ import annotations

import numpy as np

_C = np.complex128

# Parameter-free single-qubit gates.
X = np.array([[0, 1], [1, 0]], dtype=_C)
Y = np.array([[0, -1j], [1j, 0]], dtype=_C)
Z = np.array([[1, 0], [0, -1]], dtype=_C)
H = np.array([[1, 1], [1, -1]], dtype=_C) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=_C)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=_C)

# Two-qubit gates, control qubit first.
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=_C)
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=_C)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=_C)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=_C)


def rz(theta: float) -> np.ndarray:
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conjugate(e)]], dtype=_C)


def phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=_C)
