import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator

# Utility Functions
def calculate_overlaps(ansatz, prev_circuits, parameters, sampler):
    def create_fidelity_circuit(circuit_1, circuit_2):
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()
        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    overlaps = []
    for prev_circuit in prev_circuits:
        fidelity_circuit = create_fidelity_circuit(ansatz, prev_circuit)
        sampler_job = sampler.run([(fidelity_circuit, parameters)])
        meas_data = sampler_job.result()[0].data.meas
        counts_0 = meas_data.get_int_counts().get(0, 0)
        shots = meas_data.num_shots
        overlap = counts_0 / shots
        overlaps.append(overlap)

    return np.array(overlaps)

def cost_func_vqd(parameters, ansatz, prev_states, step, betas, estimator, sampler, hamiltonian):
    estimator_job = estimator.run([(ansatz, hamiltonian, [parameters])])
    total_cost = 0
    if step > 1:
        overlaps = calculate_overlaps(ansatz, prev_states, parameters, sampler)
        total_cost = np.sum([np.real(betas[state] * overlap) for state, overlap in enumerate(overlaps)])
    return estimator_job.result()[0].data.evs[0] + total_cost

# Main VQD Routine
def run_vqd(ansatz, hamiltonian, k: int = 3, betas = None, x0 = None, maxiter: int = 200):
    sampler = Sampler()
    estimator = Estimator()

    if betas is None:
        betas = [10] * k
    if x0 is None:
        x0 = np.zeros(ansatz.num_parameters)

    prev_states = []
    prev_opt_parameters = []
    eigenvalues = []

    for step in range(1, k + 1):
        if step > 1:
            prev_states.append(ansatz.assign_parameters(prev_opt_parameters))
        result = minimize(cost_func_vqd, x0,
                          args=(ansatz, prev_states, step, betas, estimator, sampler, hamiltonian),
                          method="COBYLA", options={"maxiter": maxiter})

        print(f"\nStep {step} result:")
        print(result)
        prev_opt_parameters = result.x
        eigenvalues.append(result.fun)

    return eigenvalues

# Define ansatz (2 qubits)
ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=3)

# Example Hamiltonian
observable = SparsePauliOp.from_list([("II", -2), ("XX", -1), ("YY", 2), ("ZZ", 3)])

# Run VQD to get 2 lowest eigenvalues
eigenvalues = run_vqd(ansatz, observable, k=2, betas=[50, 50])
print("\nEstimated eigenvalues:", eigenvalues)

# Convert to matrix and diagonalize
H = observable.to_matrix()
exact_eigs = np.linalg.eigvals(H)
print("Exact eigenvalues:", np.sort(np.real(exact_eigs)))