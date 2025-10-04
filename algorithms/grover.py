from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_distribution
import math
import matplotlib.pyplot as plt

# Set matplotlib style
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Grover Oracle Construction
def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)
    for target in marked_states:
        zero_inds = [ind for ind in range(num_qubits) if target[::-1].startswith("0", ind)]
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    return qc

# Build oracle
marked_states = ["0100101101"]
oracle = grover_oracle(marked_states)
oracle.draw("mpl")
plt.show()

# Build Grover operator (oracle + diffuser)
grover_op = grover_operator(oracle)
grover_op.draw("mpl")
plt.show()

# Calculate optimal number of iterations
num_qubits = grover_op.num_qubits
optimal_num_iterations = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**grover_op.num_qubits))))

# Build Grover search circuit
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
qc.measure_all()
qc.draw("mpl")
plt.show()

# Run simulation
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1).result()
counts = result[0].data["meas"].get_counts()

# Plot measurement distribution
plot_distribution(counts)
plt.tight_layout()
plt.show()