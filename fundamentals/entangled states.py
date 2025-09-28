from math import comb
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Helper: analyze a circuit
sampler = StatevectorSampler()
def analyze_state(qc: QuantumCircuit, target: np.ndarray | None, sampler: StatevectorSampler, shots: int = 1024):
    print(f"\n{qc.name.upper()} Circuit")

    # Unitary matrix
    U = Operator(qc)
    print(f"Unitary matrix of {qc.name}:\n", U.data)

    # Circuit visualization
    qc.draw("mpl")
    plt.show()

    # Measurements
    counts = sampler.run([qc.measure_all(inplace=False)], shots=shots).result()[0].data["meas"].get_counts()
    print(f"{qc.name} measurement counts: {counts}")
    plot_histogram(counts, title=f"{qc.name.upper()} Measurement")
    plt.tight_layout()
    plt.show()

    # Statevector
    state = Statevector.from_instruction(qc)
    print(f"{qc.name} statevector:", state.data)

    # Validate against target (up to global phase)
    if target is not None:
        match = np.allclose(state.data, target)# or np.allclose(state.data, -target)
        print(f"Statevector matches target (up to global phase): {match}")

    # Visualizations
    for viz in ["qsphere", "bloch", "paulivec", "hinton"]:
        print(f"Drawing {qc.name} in {viz} representation...")
        state.draw(viz)
        plt.tight_layout()
        plt.show()

# Bell states
def create_bell_state(label: str) -> QuantumCircuit:
    qc = QuantumCircuit(2, name=label)
    qc.h(0)
    qc.cx(0, 1)
    if label == "phi_minus":
        qc.z(0)
    elif label == "psi_plus":
        qc.x(1)
    elif label == "psi_minus":
        qc.x(1)
        qc.z(1)
    return qc

bell_targets = {
    "phi_plus": np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
    "phi_minus": np.array([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]),
    "psi_plus": np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
    "psi_minus": np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])}
for label in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
    qc_bell = create_bell_state(label)
    analyze_state(qc_bell, bell_targets[label], sampler)

# GHZ state (3 qubits)
def create_ghz_state() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="GHZ")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc

ghz_target = np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)])
qc_ghz = create_ghz_state()
analyze_state(qc_ghz, ghz_target, sampler)

# W state (3 qubits)
def create_w_state() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="W")
    qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
    qc.ch(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.x(0)
    return qc

w_target = np.array([0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0])
qc_w = create_w_state()
analyze_state(qc_w, w_target, sampler)

# Linear Cluster States
def create_linear_cluster(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"Cluster_{n}")
    
    # Prepare superposition on all qubits
    for i in range(n):
        qc.h(i)
    
    # CZ between neighbors
    for i in range(n - 1):
        qc.cz(i, i + 1)
    return qc

def cluster_target(n: int) -> np.ndarray:
    dim = 2 ** n
    amp = np.zeros(dim, dtype=complex)
    norm = 1 / np.sqrt(dim)
    for idx in range(dim):
        bits = [(idx >> k) & 1 for k in reversed(range(n))]
        phase = 1
        for i in range(n - 1):
            if bits[i] == 1 and bits[i + 1] == 1:
                phase *= -1
        amp[idx] = norm * phase
    return amp

qc_cluster = create_linear_cluster(4)
target_cluster = cluster_target(4)
analyze_state(qc_cluster, target_cluster, sampler)