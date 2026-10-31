"""The from-scratch simulator matches PennyLane to machine precision."""

from __future__ import annotations

import numpy as np

from src.sim.statevector import Op, run_ops
from src.sim.vs_pennylane import max_abs_diff


def test_bell_state_probabilities() -> None:
    # H on qubit 0 then CNOT(0, 1) is the Bell state (|00> + |11>)/sqrt(2).
    ops: list[Op] = [("H", (0,), ()), ("CNOT", (0, 1), ())]
    probs = run_ops(2, ops).probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


def test_known_small_circuits_match_pennylane() -> None:
    circuits: list[tuple[int, list[Op]]] = [
        (1, [("H", (0,), ())]),
        (2, [("H", (0,), ()), ("CNOT", (0, 1), ())]),
        (
            3,
            [
                ("RY", (0,), (0.3,)),
                ("RX", (1,), (1.1,)),
                ("CNOT", (1, 2), ()),
                ("RZ", (2,), (0.7,)),
                ("H", (0,), ()),
                ("CZ", (0, 2), ()),
            ],
        ),
    ]
    for n_qubits, ops in circuits:
        assert max_abs_diff(n_qubits, ops) < 1e-12


def test_random_circuits_match_pennylane() -> None:
    rng = np.random.default_rng(0)
    param_gates = ["RX", "RY", "RZ"]
    const_gates = ["H", "X", "Y", "Z", "S", "T"]
    for _ in range(5):
        n_qubits = 4
        ops: list[Op] = []
        for _ in range(20):
            kind = rng.integers(0, 3)
            if kind == 0:
                name = param_gates[rng.integers(0, len(param_gates))]
                q = int(rng.integers(0, n_qubits))
                ops.append((name, (q,), (float(rng.uniform(0, 2 * np.pi)),)))
            elif kind == 1:
                name = const_gates[rng.integers(0, len(const_gates))]
                q = int(rng.integers(0, n_qubits))
                ops.append((name, (q,), ()))
            else:
                q0, q1 = rng.choice(n_qubits, size=2, replace=False)
                gate = "CNOT" if rng.random() < 0.5 else "CZ"
                ops.append((gate, (int(q0), int(q1)), ()))
        assert max_abs_diff(n_qubits, ops) < 1e-12
