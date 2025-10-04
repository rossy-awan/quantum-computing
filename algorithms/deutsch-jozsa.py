from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Deutsch–Jozsa Oracle
def deutsch_jozsa_oracle(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits + 1)
    if np.random.randint(0, 2):
        qc.x(num_qubits)
    if np.random.randint(0, 2):
        return qc
 
    on_states = np.random.choice(range(2 ** num_qubits), 2 ** num_qubits // 2, replace=False)
    def add_cx(qc, bit_string):
        for qubit, bit in enumerate(reversed(bit_string)):
            if bit == "1":
                qc.x(qubit)
        return qc
 
    for state in on_states:
        qc = add_cx(qc, f"{state:0b}")
        qc.mcx(list(range(num_qubits)), num_qubits)
        qc = add_cx(qc, f"{state:0b}")
 
    return qc

# Build the Deutsch–Jozsa circuit
n = 3 # number of input qubits

# Create the oracle (black box)
oracle = deutsch_jozsa_oracle(n)
oracle.draw("mpl")
plt.show()

# Convert oracle to gate for use in the main circuit
blackbox = oracle.to_gate()
blackbox.label = r"$U_f$"

# Create the main Deutsch–Jozsa circuit
qc = QuantumCircuit(n + 1, n)
qc.x(n)
qc.h(range(n + 1))
qc.barrier()
qc.append(blackbox, list(range(n + 1)))
qc.barrier()
qc.h(range(n))
qc.measure(range(n), range(n))

# Draw the full circuit
qc.draw("mpl")
plt.show()

# Run the algorithm and visualize results
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1).result()
counts = result[0].data["c"].get_counts()
plot_histogram(counts)
plt.tight_layout()
plt.show()