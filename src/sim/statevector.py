"""From-scratch statevector simulator: N qubits are a 2^N complex vector.

A quantum state is a length-2^N numpy array of complex amplitudes. Gates are
unitary matrices applied by tensor-structured multiplication (never by building
the full 2^N x 2^N matrix). Measurement is the Born rule: probabilities are
squared amplitudes. Qubit 0 is the most significant bit (big-endian), matching
PennyLane's wire ordering.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.sim import gates

# A circuit is a list of operations: (gate_name, wires, params).
Op = tuple[str, Sequence[int], Sequence[float]]

_CONST_GATES = {
    "X": gates.X,
    "Y": gates.Y,
    "Z": gates.Z,
    "H": gates.H,
    "S": gates.S,
    "T": gates.T,
}
_PARAM_GATES = {"RX": gates.rx, "RY": gates.ry, "RZ": gates.rz, "PHASE": gates.phase}
_TWO_Q = {"CNOT": gates.CNOT, "CZ": gates.CZ}


class StateVector:
    """A pure state of ``n`` qubits, initialized to |0...0>."""

    def __init__(self, n_qubits: int) -> None:
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0

    @classmethod
    def zero(cls, n_qubits: int) -> StateVector:
        return cls(n_qubits)

    def apply(self, matrix: np.ndarray, wires: Sequence[int]) -> StateVector:
        """Apply a 1- or 2-qubit unitary to the given wires, in place."""
        if len(wires) == 1:
            self._apply_1q(matrix, wires[0])
        elif len(wires) == 2:
            self._apply_2q(matrix, wires[0], wires[1])
        else:
            raise ValueError(f"only 1- or 2-qubit gates supported, got {len(wires)}")
        return self

    def _apply_1q(self, matrix: np.ndarray, q: int) -> None:
        tensor = self.state.reshape([2] * self.n)
        tensor = np.tensordot(matrix, tensor, axes=([1], [q]))
        tensor = np.moveaxis(tensor, 0, q)
        self.state = tensor.reshape(-1)

    def _apply_2q(self, matrix: np.ndarray, q0: int, q1: int) -> None:
        tensor = self.state.reshape([2] * self.n)
        gate = matrix.reshape(2, 2, 2, 2)
        tensor = np.tensordot(gate, tensor, axes=([2, 3], [q0, q1]))
        tensor = np.moveaxis(tensor, [0, 1], [q0, q1])
        self.state = tensor.reshape(-1)

    def probabilities(self) -> np.ndarray:
        """Born rule: probability of each basis state is |amplitude|^2."""
        return np.abs(self.state) ** 2

    def sample(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``shots`` basis-state indices from the Born distribution."""
        probs = self.probabilities()
        return rng.choice(len(probs), size=shots, p=probs / probs.sum())


def run_ops(n_qubits: int, ops: Iterable[Op]) -> StateVector:
    """Execute a circuit (a sequence of ops) on a fresh |0...0> state."""
    sv = StateVector(n_qubits)
    for name, wires, params in ops:
        if name in _CONST_GATES:
            sv.apply(_CONST_GATES[name], wires)
        elif name in _PARAM_GATES:
            sv.apply(_PARAM_GATES[name](*params), wires)
        elif name in _TWO_Q:
            sv.apply(_TWO_Q[name], wires)
        else:
            raise ValueError(f"unknown gate: {name}")
    return sv
