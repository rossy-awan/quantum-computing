from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.circuit.library import efficient_su2
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Helper
def visualize_results(results):
    plt.figure(figsize=[5, 5])
    plt.plot(results["cost_history"], lw=1, color='k')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()

def build_callback(ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: BaseEstimatorV2, callback_dict: dict):
    def callback(current_vector):
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        
        # Compute the value of the cost function at the current vector
        current_cost = (estimator.run([(ansatz, hamiltonian, [current_vector])]).result()[0].data.evs[0])
        callback_dict["cost_history"].append(current_cost)
        
        # Print to screen on single line
        print(f"Iters. done: {callback_dict["iters"]} [Current cost: {current_cost}]")
    return callback

def cost_func(params, ansatz, hamiltonian, estimator):
    return (estimator.run([(ansatz, hamiltonian, [params])]).result()[0].data.evs[0])

def run_vqe(ansatz, hamiltonian, maxiter: int = 100, method: str = "cobyla", verbose: bool = True):
    estimator = StatevectorEstimator()
    params = 2 * np.pi * np.random.random(ansatz.num_parameters)
    callback_dict = {"prev_vector": None, "iters": 0, "cost_history": []}
    callback = build_callback(ansatz, hamiltonian, estimator, callback_dict)
    res = minimize(cost_func, x0=params, args=(ansatz, hamiltonian, estimator),
                   callback=callback, method=method, options={"maxiter": maxiter})

    if verbose:
        print(f"Estimated ground state energy: {res.fun}")
        visualize_results(callback_dict)

    return res, callback_dict

#  Hamiltonian (Heisenberg-like random field)
num_spins, ham_list = 3, []
ansatz = efficient_su2(num_qubits=num_spins, reps=3)
for i in range(num_spins):
    j = (i + 1) % num_spins # periodic connection
    ham_list.append(("XX", [i, j], .5))
    ham_list.append(("YY", [i, j], .5))
    ham_list.append(("ZZ", [i, j], .5))
# for i in range(num_spins):
#     ham_list.append(("X", [i], np.random.random() * 2 - 1))
#     ham_list.append(("Y", [i], np.random.random() * 2 - 1))
#     ham_list.append(("Z", [i], np.random.random() * 2 - 1))
hamiltonian = SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_spins)
# run_vqe(ansatz, hamiltonian)

# Electronic ground state energy of H2
num_spins = 1
ansatz = efficient_su2(num_qubits=num_spins, reps=1)
ham_list = [('I', -1.04886087), ('X', 0.18121804), ('Z', -0.7967368)]
h2_hamiltonian = SparsePauliOp.from_list(ham_list, num_qubits=num_spins)
eigenvalues, eigenvectors = np.linalg.eig(np.array(h2_hamiltonian))
run_vqe(ansatz, h2_hamiltonian)
print("Electronic ground state energy (Hartree): ", min(eigenvalues).real)