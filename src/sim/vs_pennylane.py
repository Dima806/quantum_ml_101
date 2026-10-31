"""Validate the from-scratch simulator against PennyLane to machine precision.

Both run the identical circuit spec and we compare final statevectors. If the
linear algebra is right, they agree to ~1e-15.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pennylane as qml

from src.sim.statevector import Op, run_ops

_CONST = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
    "H": qml.Hadamard,
    "S": qml.S,
    "T": qml.T,
}
_PARAM = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ, "PHASE": qml.PhaseShift}


def _apply(name: str, wires: Sequence[int], params: Sequence[float]) -> None:
    if name in _CONST:
        _CONST[name](wires=wires[0])
    elif name in _PARAM:
        _PARAM[name](params[0], wires=wires[0])
    elif name == "CNOT":
        qml.CNOT(wires=list(wires))
    elif name == "CZ":
        qml.CZ(wires=list(wires))
    else:
        raise ValueError(f"unknown gate: {name}")


def pennylane_state(n_qubits: int, ops: Iterable[Op]) -> np.ndarray:
    """Run the circuit on PennyLane's default.qubit and return the statevector."""
    dev = qml.device("default.qubit", wires=n_qubits)
    ops_list = list(ops)

    @qml.qnode(dev)
    def circuit():
        for name, wires, params in ops_list:
            _apply(name, wires, params)
        return qml.state()

    return np.asarray(circuit())


def max_abs_diff(n_qubits: int, ops: Iterable[Op]) -> float:
    """Largest absolute amplitude difference between the two simulators."""
    ops_list = list(ops)
    ours = run_ops(n_qubits, ops_list).state
    theirs = pennylane_state(n_qubits, ops_list)
    return float(np.max(np.abs(ours - theirs)))
