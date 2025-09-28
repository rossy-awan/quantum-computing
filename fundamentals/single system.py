from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Build a single-qubit circuit and add a Hadamard gate
qc = QuantumCircuit(1)

# Put the qubit into a superposition
qc.h(0)

# Visualize the circuit
qc.draw("mpl")
plt.show()

# Run measurements using the Sampler primitive
sampler = StatevectorSampler()
job = sampler.run([qc.measure_all(inplace=False)], shots=1000)
result = job.result()

# Retrieve measurement counts
counts = result[0].data["meas"].get_counts()
print(f"Measurement counts: {counts}")

# Plot measurement histogram
plot_histogram(counts)
plt.tight_layout()
plt.show()

# Get the statevector and visualize it
state = Statevector.from_instruction(qc)

# Draw different visual representations of the state
for output in ["qsphere", "bloch", "paulivec", "hinton"]:
    print(f"Drawing state in {output} representation...")
    state.draw(output)
    plt.tight_layout()
    plt.show()

# Display the unitary operator of the circuit
U = Operator(qc)
print(f"Unitary matrix of the circuit (Hadamard):\n{U.data}")