from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, thermal_relaxation_error
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})


# Add depolarizing error to all single qubit u1, u2, u3 gates on qubit 0 only
noise_model = NoiseModel()
noise_model.add_quantum_error(depolarizing_error(.05, 1), ["u1", "u2", "u3"], [0])
print(noise_model)

# Add depolarizing error to all single qubit u1, u2, u3 gates
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(.05, 1), ["u1", "u2", "u3"])
print(noise_model)


# System Specification
n_qubits = 4
qc = QuantumCircuit(n_qubits)

# Test Circuit
qc.h(0)
for qubit in range(n_qubits - 1):
    qc.cx(qubit, qubit + 1)
qc.measure_all()
qc.draw('mpl')
plt.show()

# Ideal simulator and execution
result_ideal = AerSimulator().run(qc).result()
plot_histogram(result_ideal.get_counts(0))
plt.tight_layout()
plt.show()


# Example error probabilities
p_reset = .01
p_meas = .03
p_gate1 = .02

# QuantumError objects
bit_flip = lambda p: pauli_error([("X", p), ("I", 1 - p)])
phase_flip = lambda p: pauli_error([("Z", p), ("I", 1 - p)])
error = lambda p: bit_flip(p).compose(phase_flip(p))
error_reset = error(p_reset)
error_meas = error(p_meas)
error_gate1 = error(p_gate1)
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_reset, "reset")
noise_model.add_all_qubit_quantum_error(error_meas, "measure")
noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])
print(noise_model)

# Noisy simulator and execution
sim_noise = AerSimulator(noise_model=noise_model)
passmanager = generate_preset_pass_manager(optimization_level=0, backend=sim_noise)
result_noise = sim_noise.run(passmanager.run(qc)).result()
plot_histogram(result_noise.get_counts(0))
plt.show()


# T1 and T2 values for qubits 0-3
T1s = np.random.normal(50e3, 10e3, 4) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(60e3, 10e3, 4) # Sampled from normal distribution mean 60 microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

# Instruction times (in nanoseconds)
time_u1 = 0 # virtual gate
time_u2 = 50 # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000 # 1 microsecond
time_measure = 1000 # 1 microsecond

# QuantumError objects
error = lambda t: [thermal_relaxation_error(t1, t2, t) for t1, t2 in zip(T1s, T2s)]
errors_reset = error(time_reset)
errors_measure = error(time_measure)
errors_u1 = error(time_u1)
errors_u2 = error(time_u2)
errors_u3 = error(time_u3)
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)] for t1b, t2b in zip(T1s, T2s)]

# Add errors to noise model
noise_thermal = NoiseModel()
for j in range(4):
    noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(4):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
print(noise_thermal)

# Run the noisy simulation
sim_thermal = AerSimulator(noise_model=noise_thermal)
passmanager = generate_preset_pass_manager(optimization_level=0, backend=sim_thermal)
result_thermal = sim_thermal.run(passmanager.run(qc)).result()
plot_histogram(result_thermal.get_counts(0))
plt.tight_layout()
plt.show()