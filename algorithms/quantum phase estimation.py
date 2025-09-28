from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Helper
def run_qpe(unitary: QuantumCircuit, n_estimation: int, eigenstate: QuantumCircuit, shots: int = 1024, draw: bool = True) -> tuple[float, dict]:
    n_target = unitary.num_qubits

    # QPE circuit
    qpe = PhaseEstimation(n_estimation, unitary)

    # Classical and quantum register
    qreg = QuantumRegister(n_estimation + n_target, "q")
    creg = ClassicalRegister(n_estimation, "c")
    qc = QuantumCircuit(qreg, creg)

    # Compose
    qc.compose(eigenstate, qubits=range(n_estimation, n_estimation + n_target), inplace=True)
    qc.compose(qpe, qubits=range(n_estimation + n_target), inplace=True)

    # Run measurements using the Sampler primitive
    qc.measure(range(n_estimation), range(n_estimation))
    counts = StatevectorSampler().run([qc], shots=shots).result()[0].data['c'].get_counts()
    bitstring = max(counts, key=counts.get)[::-1] # reverse bit order
    phase_est = int(bitstring, 2) / (2 ** n_estimation)

    if draw:
        print(f"Most likely bits: {bitstring} → {phase_est}")
        qc.draw("mpl"); plt.show()
        plot_histogram(counts); plt.show()

    return phase_est, counts

# Example
for i in range(10):
    theta = np.random.random()
    U = QuantumCircuit(1)
    U.rz(4 * np.pi * theta, 0)
    state = QuantumCircuit(1)
    state.x(0)
    phase, counts = run_qpe(U, 8, state)
    print(f"Estimate: {round(phase, 4)} Theory: {round(theta, 4)}")