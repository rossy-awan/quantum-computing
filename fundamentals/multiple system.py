from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})
sampler = StatevectorSampler()

# Utilities
def analyze_state(qc: QuantumCircuit, sampler: StatevectorSampler, target: np.ndarray = None, shots: int = 1024, title: str = None):
    name = qc.name if title is None else title
    print(f"\n{name.upper()} Circuit")

    # Unitary matrix
    U = Operator(qc)
    print(f"Unitary matrix of {name}:\n", U.data)

    # Circuit visualization
    qc.draw("mpl")
    plt.show()

    # Measurements
    counts = sampler.run([qc.measure_all(inplace=False)], shots=shots).result()[0].data["meas"].get_counts()
    print(f"{name} measurement counts: {counts}")
    plot_histogram(counts, title=f"{name} Measurement")
    plt.tight_layout()
    plt.show()

    # Statevector
    state = Statevector.from_instruction(qc)
    print(f"{name} statevector:", state.data)

    # Validate against target (up to global phase)
    if target is not None:
        print(f"Statevector matches target (up to global phase): {np.allclose(state.data, target)}")

    # State visualizations
    for viz in ["qsphere", "bloch", "paulivec", "hinton"]:
        print(f"Drawing {name} in {viz} representation...")
        state.draw(viz)
        plt.tight_layout()
        plt.show()

# Product / separable state
def create_product_state() -> QuantumCircuit:
    qc = QuantumCircuit(2, name="ProductState")
    qc.h(0)
    return qc
product_qc = create_product_state()
target_product = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
analyze_state(product_qc, sampler, target=target_product, title="Product State")

# Partial entanglement
def create_partial_entanglement() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="PartialEntanglement")
    qc.h(0)
    qc.cx(0, 1)
    return qc
partial_qc = create_partial_entanglement()
target_partial = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0, 0, 0, 0])
analyze_state(partial_qc, sampler, target=target_partial, title="Partial Entanglement")