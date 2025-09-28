from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from qiskit.primitives import StatevectorSampler
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})
sampler = StatevectorSampler()

# Helper
def analyze_gate(qc: QuantumCircuit, name: str):
    print(f"\n{name}")
    
    # Statevector & Unitary
    state = Statevector.from_instruction(qc)
    print("Statevector:", state.data)
    print("Unitary:\n", Operator(qc).data)

    # Circuit
    qc.draw("mpl")
    plt.show()

    # Measurement
    counts = sampler.run([qc.measure_all(inplace=False)], shots=1024).result()[0].data["meas"].get_counts()
    print("Measurement counts:", counts)
    plot_histogram(counts, title=f"{name} measurement")
    plt.tight_layout()
    plt.show()

# Define a list of multi-qubit gates to explore
multi_gates = [
    # 2-qubit gates
    ("CX (CNOT)",          lambda qc: qc.cx(0, 1), 2),
    ("CY",                 lambda qc: qc.cy(0, 1), 2),
    ("CZ",                 lambda qc: qc.cz(0, 1), 2),
    ("CH",                 lambda qc: qc.ch(0, 1), 2),
    ("CRX(pi/4)",          lambda qc: qc.crx(np.pi/4, 0, 1), 2),
    ("CRY(pi/4)",          lambda qc: qc.cry(np.pi/4, 0, 1), 2),
    ("CRZ(pi/4)",          lambda qc: qc.crz(np.pi/4, 0, 1), 2),
    ("CS",                 lambda qc: qc.cs(0, 1), 2),
    ("CU(pi/3, 0, pi/2)",  lambda qc: qc.cu(np.pi/3, 0, np.pi/2, 0, 0, 1), 2),
    ("SWAP",               lambda qc: qc.swap(0, 1), 2),
    ("iSWAP",              lambda qc: qc.iswap(0, 1), 2),
    ("ECR",                lambda qc: qc.ecr(0, 1), 2),
    ("RXX(pi/3)",          lambda qc: qc.rxx(np.pi/3, 0, 1), 2),
    ("RYY(pi/3)",          lambda qc: qc.ryy(np.pi/3, 0, 1), 2),
    ("RZZ(pi/3)",          lambda qc: qc.rzz(np.pi/3, 0, 1), 2),
    ("RZX(pi/3)",          lambda qc: qc.rzx(np.pi/3, 0, 1), 2),

    # 3-qubit gates
    ("Toffoli (CCX)",      lambda qc: qc.ccx(0, 1, 2), 3),
    ("CSWAP (Fredkin)",    lambda qc: qc.cswap(0, 1, 2), 3),
    ("CCZ",                lambda qc: qc.ccz(0, 1, 2), 3),

    # 4-qubit example (multi-controlled X)
    ("C3X",                lambda qc: qc.mcx([0, 1, 2], 3), 4),
]

# Explore each gate
for name, apply_gate, n_qubits in multi_gates:
    qc = QuantumCircuit(n_qubits, name=name)
    qc.x(0)
    apply_gate(qc)
    analyze_gate(qc, name)