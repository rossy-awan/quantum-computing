from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Time evolution of Spin
def trotter_evolution(hamiltonian, observables, time: float, steps: int,
                      state: QuantumCircuit | np.ndarray | None = None,
                      order: int = 2, precision: float = 1e-6):
    n = hamiltonian.num_qubits
    t, dt = np.linspace(0, time, steps, retstep=True)
    estimator = StatevectorEstimator()

    # Prepare initial circuit
    if state is None:
        qc_init = QuantumCircuit(n)
    elif isinstance(state, QuantumCircuit):
        qc_init = state
    elif isinstance(state, np.ndarray):
        sv = Statevector(state)
        qc_init = QuantumCircuit(n)
        qc_init.append(sv.to_instruction(), qc_init.qubits)
    
    # Pre-build single-step evolution gate
    evolution_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=SuzukiTrotter(order=order))

    # Time evolution quantum circuit
    qc = QuantumCircuit(n)
    qc.compose(qc_init, inplace=True)

    # Storage for results
    jobs = []
    results = {label: [] for label in observables.keys()}

    # Build all jobs in one pass
    for step in range(len(t)):
        qc.append(evolution_gate, qc.qubits)
        snap = qc.copy()
        for obs in observables.values():
            jobs.append((snap, obs))

    # Run all jobs at once
    results_raw = estimator.run(jobs, precision=precision).result()

    # Collect outputs
    num_obs = len(observables)
    chunks = [results_raw[i:i+num_obs] for i in range(0, len(results_raw), num_obs)]
    for step, chunk in enumerate(chunks):
        for label, raw in zip(observables.keys(), chunk):
            results[label].append(raw.data.evs.item())

    return t, results

# Example: 3-spin Heisenberg-like model (PBC)
n = 3

# Coupling constants per edge (site-dependent)
Jx = {(0,1): 1.0, (1,2): 0.9, (2,0): 1.2}
Jy = {(0,1): 0.8, (1,2): 1.1, (2,0): 0.7}
Jz = {(0,1): 1.2, (1,2): 1.3, (2,0): 0.9}

# Local fields per site
h = {0: 0.5, 1: -0.3, 2: 0.2}

# Build Heisenberg couplings
terms, coeffs = [], []
for (i, j), J in Jx.items():
    pauli = ["I"] * n
    pauli[i] = "X"
    pauli[j] = "X"
    terms.append("".join(pauli))
    coeffs.append(J)

for (i, j), J in Jy.items():
    pauli = ["I"] * n
    pauli[i] = "Y"
    pauli[j] = "Y"
    terms.append("".join(pauli))
    coeffs.append(J)

for (i, j), J in Jz.items():
    pauli = ["I"] * n
    pauli[i] = "Z"
    pauli[j] = "Z"
    terms.append("".join(pauli))
    coeffs.append(J)

# Add local fields
for i, hi in h.items():
    pauli = ["I"] * n
    pauli[i] = "Z"
    terms.append("".join(pauli))
    coeffs.append(hi)

hamiltonian = SparsePauliOp(terms, coeffs)

# Observables of interest
observables = {
    "ZZI": SparsePauliOp("ZZI"),
    "IZZ": SparsePauliOp("IZZ"),
    "ZIZ": SparsePauliOp("ZIZ"),
}

# Run trotter evolution
times, results = trotter_evolution(hamiltonian, observables, time=5, steps=200, order=1)

plt.figure(figsize=(5, 5))
for label, values in results.items():
    plt.plot(times, values, label=label)
plt.title("3-spin Heisenberg model")
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle\hat{O}\rangle$")
plt.legend()
plt.tight_layout()
plt.show()