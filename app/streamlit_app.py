"""Streamlit app: statevector explorer, the arena, the barren-plateau meter,
and the 2^N simulation wall. All CPU, all local, no account, no hardware.

Run with: make run
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis.barren import gradient_variance
from src.analysis.scaling import statevector_bytes
from src.datasets.toy import get_dataset
from src.evaluation.comparison import compare
from src.sim import gates
from src.sim.statevector import StateVector

_SINGLE = {"H": gates.H, "X": gates.X, "Y": gates.Y, "Z": gates.Z}


def statevector_explorer() -> None:
    st.header("Statevector explorer")
    st.caption("A quantum state is just 2^N complex numbers. Add gates and watch it.")
    n_qubits = st.slider("qubits", 1, 5, 2)
    sv = StateVector(n_qubits)
    for q in range(n_qubits):
        gate = st.selectbox(f"gate on qubit {q}", ["(none)", *_SINGLE], key=f"g{q}")
        if gate in _SINGLE:
            sv.apply(_SINGLE[gate], (q,))
    if n_qubits > 1 and st.checkbox("entangle (CNOT 0->1)"):
        sv.apply(gates.CNOT, (0, 1))
    probs = sv.probabilities()
    labels = [format(i, f"0{n_qubits}b") for i in range(len(probs))]
    st.bar_chart(pd.DataFrame({"probability": probs}, index=labels))


def arena() -> None:
    st.header("Quantum vs classical arena")
    name = st.selectbox("dataset", ["moons", "circles", "iris", "planted"])
    if st.button("Run arena"):
        X, y = get_dataset(name, seed=0)
        with st.spinner("fitting models..."):
            results = compare(X, y, seed=0)
        frame = pd.DataFrame(
            {
                "model": [r.name for r in results],
                "accuracy": [r.accuracy for r in results],
                "fit_seconds": [r.fit_seconds for r in results],
            }
        ).sort_values("accuracy", ascending=False)
        st.dataframe(frame, hide_index=True)
        st.caption("On ordinary data the classical baselines usually win.")


def barren_meter() -> None:
    st.header("Barren-plateau meter")
    n_qubits = st.slider("qubits", 2, 7, 4)
    n_layers = st.slider("depth (layers)", 2, 20, 10)
    variance = gradient_variance(n_qubits, n_layers, n_samples=80, seed=0)
    st.metric("Var(dC/dtheta)", f"{variance:.2e}")
    st.caption("Slide qubits up and watch the gradient variance collapse toward zero.")


def simulation_wall() -> None:
    st.header("The 2^N simulation wall")
    max_qubits = st.slider("max qubits", 4, 40, 30)
    counts = list(range(1, max_qubits + 1))
    frame = pd.DataFrame(
        {"bytes": [statevector_bytes(n) for n in counts]},
        index=counts,
    )
    st.line_chart(frame)
    st.caption("~16 MB at 20 qubits, ~1 GB at 26, more than exists on Earth at 40.")


def main() -> None:
    st.set_page_config(page_title="quantum_ml_101", layout="wide")
    st.title("Quantum Machine Learning 101")
    page = st.sidebar.radio(
        "view",
        ["Statevector explorer", "Arena", "Barren-plateau meter", "Simulation wall"],
    )
    if page == "Statevector explorer":
        statevector_explorer()
    elif page == "Arena":
        arena()
    elif page == "Barren-plateau meter":
        barren_meter()
    else:
        simulation_wall()


if __name__ == "__main__":
    main()
