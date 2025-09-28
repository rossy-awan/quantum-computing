from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms import TimeEvolutionProblem, VarQRTE
from qiskit_algorithms.time_evolvers.variational import RealMcLachlanPrinciple
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Helper
def run_var_qrte(hamiltonian, observables, reps: int = 1, init_angle: float = np.pi/2, time: float = 1):
    
    # Ansatz
    ansatz = efficient_su2(hamiltonian.num_qubits, reps=reps)
    init_param_values = np.ones(len(ansatz.parameters)) * init_angle
    
    # Evolution problem
    problem = TimeEvolutionProblem(hamiltonian, time, aux_operators=magnetization)

    # Variational QRTE
    principle = RealMcLachlanPrinciple()
    qc = VarQRTE(ansatz, init_param_values, principle, StatevectorEstimator())
    
    # Observables
    result = qc.evolve(problem)
    times = result.times
    observables = np.array([[val[0] for val in obs] for obs in result.observables])
    
    return times, observables, ansatz

# Example: 3-spin Heisenberg-like model (PBC)
hamiltonian = SparsePauliOp.from_list([("XXI", 1.82),
                                       ("IXX", 1.77),
                                       ("XIX", 1.93),
                                       ("YYI", 2.75),
                                       ("IYY", 1.86),
                                       ("YIY", 2.64),
                                       ("ZZI", 1.91),
                                       ("IZZ", 2.73),
                                       ("ZIZ", 1.89),
                                       ("ZII", -.32),
                                       ("IZI", -.28),
                                       ("IIZ", -.25),])
magnetization = [Pauli("XII"), Pauli("IXI"), Pauli("IIX"),
                 Pauli("YII"), Pauli("IYI"), Pauli("IIY"),
                 Pauli("ZII"), Pauli("IZI"), Pauli("IIZ")]

times, observables, ansatz = run_var_qrte(hamiltonian, magnetization)

plt.figure(figsize=(5, 5))
plt.plot(times, observables[:, 0], label=r"$\langle X_1 \rangle$", lw=1)
plt.plot(times, observables[:, 1], label=r"$\langle X_2 \rangle$", lw=1)
plt.plot(times, observables[:, 2], label=r"$\langle X_3 \rangle$", lw=1)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle X_i \rangle$")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(times, observables[:, 3], label=r"$\langle Y_1 \rangle$", lw=1)
plt.plot(times, observables[:, 4], label=r"$\langle Y_2 \rangle$", lw=1)
plt.plot(times, observables[:, 5], label=r"$\langle Y_3 \rangle$", lw=1)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle Y_i \rangle$")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(times, observables[:, 6], label=r"$\langle Z_1 \rangle$", lw=1)
plt.plot(times, observables[:, 7], label=r"$\langle Z_2 \rangle$", lw=1)
plt.plot(times, observables[:, 8], label=r"$\langle Z_3 \rangle$", lw=1)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle Z_i \rangle$")
plt.legend()
plt.tight_layout()
plt.show()