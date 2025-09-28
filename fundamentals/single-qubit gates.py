from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Helper: visualize the circuit and draw state on Bloch sphere
def visualize(qc: QuantumCircuit, state: Statevector):
    # Bloch sphere
    state.draw("bloch")
    plt.tight_layout()
    
    # Circuit
    qc.draw("mpl")
    plt.show()

# Define a list of single-qubit gates to explore
gates = [
    ("X",  lambda qc: qc.x(0)),           # Pauli-X
    ("Y",  lambda qc: qc.y(0)),           # Pauli-Y
    ("Z",  lambda qc: qc.z(0)),           # Pauli-Z
    ("H",  lambda qc: qc.h(0)),           # Hadamard
    ("S",  lambda qc: qc.s(0)),           # Phase (sqrt(Z))
    ("Sdg",lambda qc: qc.sdg(0)),         # S dagger
    ("T",  lambda qc: qc.t(0)),           # π/8 gate
    ("Tdg",lambda qc: qc.tdg(0)),         # T dagger
    ("Rx", lambda qc: qc.rx(np.pi/2, 0)), # π/2 rotation about X
    ("Ry", lambda qc: qc.ry(np.pi/2, 0)), # π/2 rotation about Y
    ("Rz", lambda qc: qc.rz(np.pi/2, 0)), # π/2 rotation about Z
]

# Explore each gate: statevector & Bloch
for name, gate in gates:
    qc = QuantumCircuit(1)
    gate(qc)
    state = Statevector.from_instruction(qc)
    print(f"\nGate: {name}")
    print(f"\nStatevector:\n{state.data}")
    print(f"\nUnitary:\n{Operator(qc).data}")
    visualize(qc, state)

# Helper function to measure in a chosen basis
def measure_in_basis(circuit: QuantumCircuit, basis: str, shots: int = 1024):
    qc = circuit.copy()

    # Apply basis change before measuring
    if basis.upper() == "X":
        qc.h(0)
    elif basis.upper() == "Y":
        qc.sdg(0)
        qc.h(0)

    # Visualize the circuit
    qc.draw("mpl")
    plt.show()

    # Run measurements using the statevector Sampler
    sampler = StatevectorSampler()
    counts = sampler.run([qc.measure_all(inplace=False)], shots=shots).result()[0].data["meas"].get_counts()

    # Plot histogram
    plot_histogram(counts)
    plt.title(f"Measurement in {basis.upper()} basis")
    plt.tight_layout()
    plt.show()

    return counts

# Example: Prepare quantum circuit and measure in Z, X, Y bases
qc = QuantumCircuit(1)
qc.h(0)

# Measurement in Different Bases (Z, X, Y)
print("\nMeasurement in different bases")
counts_x = measure_in_basis(qc, "X")
print("X-basis counts:", counts_x)
counts_y = measure_in_basis(qc, "Y")
print("Y-basis counts:", counts_y)
counts_z = measure_in_basis(qc, "Z")
print("Z-basis counts:", counts_z)