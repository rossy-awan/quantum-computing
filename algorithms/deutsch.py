from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Deutsch algorithm implementation in Qiskit
def twobit_function(case: int):
    if case not in [1, 2, 3, 4]:
        raise ValueError("`case` must be 1, 2, 3, or 4.")
    f = QuantumCircuit(2, name="$U_f$")

    # Apply CX if the function depends on the input
    if case in [2, 3]:
        f.cx(0, 1)

    # Apply X if the function flips the output
    if case in [3, 4]:
        f.x(1)

    return f

# Choose which oracle (black box) you want to test
blackbox = twobit_function(3).to_gate() # Choose 1, 2, 3, or 4
blackbox.label = r"$U_f$"

# Build the Deutsch circuit
qc = QuantumCircuit(2, 1)
qc.x(1)
qc.h(range(2))
qc.barrier(label="Oracle")
qc.append(blackbox, [0, 1])
qc.barrier()
qc.h(0)
qc.measure(0, 0)

# Draw the circuit
qc.draw("mpl")
plt.show()

# Run the circuit on the StatevectorSampler
sampler = StatevectorSampler()
counts = sampler.run([qc], shots=1).result()[0].data["c"].get_counts()

# Plot the histogram of the measurement result
plot_histogram(counts)
plt.tight_layout()
plt.show()