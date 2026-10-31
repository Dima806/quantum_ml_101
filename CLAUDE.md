# quantum_ml_101

Hype-free, hands-on quantum machine learning that runs **entirely on CPU** with free,
keyless, open-source simulators. No quantum hardware, no cloud, no account, no API key.

Deliverable shape: a GitHub repo with **6 notebooks**, a **`src/` library**, and a
**Streamlit app**, all runnable inside a **2-CPU / 8 GB GitHub Codespace**. Full spec: [.llm/PRD_quantum_ml_101.md](.llm/PRD_quantum_ml_101.md).

## Thesis — three claims, each one measured, not asserted

1. **A quantum circuit is linear algebra.** N qubits = a `2^N` vector of complex amplitudes;
   each gate is a unitary matrix multiplying it; measurement is the Born rule (`|amp|^2`). We
   write the simulator from scratch in numpy and it agrees with PennyLane **to machine precision**.
2. **Classical wins on runnable data.** Quantum-kernel SVMs and variational quantum circuits
   (VQCs) are matched or beaten by logistic regression / RBF-SVM / a small MLP, at a fraction of
   the cost. The one case where a quantum kernel "wins" is a dataset **engineered** for its
   feature map — shown, and **flagged honestly as a caveat, never as a triumph**.
3. **The barren plateau is why deep VQCs don't train.** `Var(∂C/∂θ)` for random deep circuits
   collapses **exponentially** with qubit count until the gradient is numerically zero everywhere.
   A proven landscape property, not a bug or a tuning problem.

The honest verdict is neither hype nor dismissal. Every quantum model is put in an **arena**
next to a real classical baseline on accuracy and runtime. Never ship a quantum result without
its baseline beside it.

## Hard constraints — this IS the point, do not relax

- **All classical simulation of SMALL circuits.** NO quantum hardware / QPU, NO cloud, NO
  account, NO API key, NO registration, NO Docker, NO GPU. If a task would need any of these,
  it is out of scope — say so rather than reaching for it.
- **Keyless & local.** PennyLane + `pennylane-lightning` run locally with no account. Qiskit +
  Aer are an equally keyless local alternative (optional `qiskit` extra). The from-scratch
  simulator needs **numpy only**. Real-hardware access would need an account → deliberately out
  of scope, and named as a separate world in the framework notebook.
- **Stay under the ceiling.** A statevector is `2^N` complex128 (16 bytes each): ~16 MB @ 20q,
  ~1 GB @ 26q. **Demos cap at ~16–18 qubits.** The `2^N` memory/time curve is itself a lesson —
  teach the wall, don't crash into it.
- **Python 3.11+, `uv` for everything.** Each notebook must execute in **< 5 minutes** on 2 CPUs.
  (uv currently resolves Python 3.14 / numpy 2.5 / PennyLane 0.45 here; the `>=` floors still hold.)

## Repository map

```
config/settings.yaml          qubit counts, circuit depths, datasets, seeds
src/
  config.py                   pydantic-settings loader for settings.yaml
  sim/
    statevector.py            from-scratch numpy statevector sim (state, apply gate, measure)
    gates.py                  single-/two-qubit unitaries (X, H, RY, CNOT, ...)
    vs_pennylane.py           validate from-scratch sim vs PennyLane to machine precision
  qml/
    feature_maps.py           angle + ZZ feature maps (encode data -> state)
    kernels.py                quantum kernel |<phi(x)|phi(x')>|^2 -> matrix -> sklearn SVM
    vqc.py                    variational classifier: ansatz + parameter-shift gradient
    train.py                  VQC training loop (gradient descent)
  classical/baselines.py      logistic regression, RBF-SVM, small MLP — the yardstick
  analysis/
    barren.py                 gradient variance vs qubits/depth (the plateau)
    scaling.py                memory & time vs qubit count (the 2^N wall)
  datasets/toy.py             moons, circles, iris-2d + planted set (Havlíček ad-hoc, favors ZZ kernel)
  evaluation/
    arena.py                  quantum vs classical: accuracy + runtime
    comparison.py             harness
  visualisation.py            shared plotting (matplotlib + plotly)
app/streamlit_app.py          4 tabs: statevector explorer, arena, barren meter, sim wall
notebooks/01..06_*.ipynb      see "Notebooks" below
tests/                        test_statevector, test_kernel, test_vqc, test_barren
outputs/figures/*.png         every figure (notebooks save here)
outputs/data/*.json           every number & caption (notebooks save here; NB 06 reads them back)
```

## Core math — keep the code honest

- **State:** `psi` is a length-`2^N` complex128 numpy array, normalized (`sum|amp|^2 == 1`).
  Index the computational basis in big-endian qubit order; pick one convention and keep it
  everywhere (sim, feature maps, PennyLane validation).
- **Gates:** unitary matrices applied with tensor-structured multiplication (reshape to a rank-N
  tensor and contract the target axis), NOT by materializing the full `2^N × 2^N` matrix.
- **Measurement:** Born rule — probabilities are squared amplitudes.
- **Entanglement:** a Bell state's correlations cannot factor into single-qubit states; demo
  this explicitly (it is a success criterion).
- **Quantum kernel:** `K[i,j] = |<phi(x_i)|phi(x_j)>|^2`. Must come out symmetric and PSD, and
  plug straight into `sklearn.svm.SVC(kernel="precomputed")` — the `kernels_101` trick unchanged.
- **VQC:** parameterized unitary trained by gradient descent; gradient via the **parameter-shift
  rule** (exact), the quantum analog of backprop. Always cross-check against finite differences.
- **Barren plateau:** sample random params for random deep ansätze, measure `Var(∂C/∂θ)` across
  the landscape, show it halving per added qubit.

## Conventions

- **Config, not constants.** Qubit counts, depths, dataset choices, and seeds live in
  `config/settings.yaml`, loaded via `src/config.py` (pydantic / pydantic-settings). Don't
  hard-code these in notebooks or `src/`.
- **Seed everything.** Datasets, random circuit params, and train/test splits are seeded so
  every figure and assertion is reproducible. Datasets are small, seeded, generated on the fly.
- **Imports** use the `src` package: `from src.sim.statevector import ...`. The project is a
  virtual (non-installed) uv project; notebooks and the app run from the repo root.
- **Packaging/tooling:** `[tool.uv] package = false` (`uv sync` installs deps, no build backend);
  tests get the repo root via pytest `pythonpath = ["."]`. `ruff` (format + lint, line-length 99,
  rules `E F W I UP N B A SIM PTH`, with `N802/N803/N806` ignored so math/ML names like `X`, `C`,
  `U` are allowed), `ty` type-checks `src/`, `pytest` runs tests. `make lint` and `make test`
  must be **zero errors** before calling work done.
- **Notebook outputs.** Every notebook saves figures to `outputs/figures/*.png` and all
  textual/numerical results to `outputs/data/*.json` via a `save_json` helper; NB 06 reads those
  JSON files back to build the summary figure. The setup cell anchors outputs to the repo root
  and runs on the `quantum-ml-101` kernel (registered by `make setup`). `make notebooks` leaves
  `*.nbconvert.ipynb` smoke-test copies (gitignored).
- **Planted dataset.** `datasets.toy.planted_quantum` is the Havlíček *ad-hoc* construction:
  labels are the sign of `<phi(x)|M|phi(x)>` for a fixed random observable `M`, a rule linear in
  the fidelity-kernel feature space, so the ZZ quantum kernel separates it by design. It is the
  *engineered exception* — always flag it as such, never as a general advantage.
- **Plotting** goes through `src/visualisation.py` for a consistent look; notebooks import it
  rather than re-styling. All figures land in `outputs/figures/`.

## Correctness bars (these are asserted in tests, not just shown)

- `test_statevector.py` — from-scratch sim matches PennyLane to machine precision.
- `test_kernel.py` — quantum kernel symmetric + PSD, plugs into sklearn SVM.
- `test_vqc.py` — parameter-shift gradient matches finite differences.
- `test_barren.py` — gradient variance decreases with qubit count.

Other graded outcomes: the `2^N` scaling curve is measured; classical matches/beats quantum on
ordinary data; the quantum "win" appears only on the planted set and is flagged.

**Measured (latest full run):** sim vs PennyLane `0.0`; Bell purity `0.5`; memory `16.8 MB @ 20q`
/ `1.07 GB @ 26q`; param-shift vs finite-diff `~2e-9`; planted set ZZ kernel `0.94` vs best
classical `0.81`; barren-plateau variance `×0.52` per qubit (`54×` over 2→8q). `make notebooks`
runs all six in ~2.5 min total.

## Notebooks (narrative order; each is a build + a measurement)

1. **A Qubit Is a Vector** — demystify: qubit = 2-vector, gate = matrix, measurement = Born rule;
   build the 1–2 qubit sim, validate vs PennyLane. Install the free keyless tools.
2. **Circuits and Entanglement** — `2^N` via tensor products, Bell-state entanglement, and the
   memory/time doubling plotted (the link to `curse_of_dimensionality_101`).
3. **Feature Maps and Kernels** (showpiece 1) — angle/ZZ encodings, quantum kernel → SVM vs
   RBF-SVM; usual result + the engineered exception, flagged.
4. **Variational Circuits** (showpiece 2) — ansatz + parameter-shift training vs logistic/MLP
   (the link to `backprop_101` / `gradient_descent_101`).
5. **The Barren Plateau** (the crux) — gradient-variance collapse measured; mitigations named
   (shallow circuits, local cost, structured ansätze) honestly.
6. **Decision Framework** — what's real vs hype, always run the baseline, the CPU qubit ceiling,
   the keyless software landscape, and the barren plateau as the measured reason not to scale.

## Commands

```
make setup       first-time: install uv, uv sync --all-extras, register ipykernel
make lint        ruff format + ruff check --fix + ty check   (zero errors required)
make test        pytest            make test-cov for coverage
make notebooks   execute all notebooks (<5 min each, 300s timeout)
make run         streamlit app on :8501
make lab         jupyterlab on :8888
make ci          sync + lint + test        make dev   fast lint + test loop
```

## Working rules

- **Caveman mode: ON.** Code-first, minimal prose. `/compact` at notebook boundaries.
- When unsure about a quantum/ML fact, **measure it** in code rather than asserting — that is the
  whole ethos of this repo.
