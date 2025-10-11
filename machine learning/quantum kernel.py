import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UnitaryOverlap, ZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_distribution

plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Two mock data points (last element = class label)
def generate_mock_data(num_points=2, num_features=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(loc=0.0, scale=0.2, size=(num_points, num_features))
    y = np.array([[1], [-1]])
    data = np.hstack([X, y])
    return data.tolist()
small_data = np.round(generate_mock_data(seed=42), 2)
print(small_data)

# Remove labels for feature mapping
train_data = [d[:-1] for d in small_data]
feature_dim = len(train_data[0])

# Create Feature Map and Overlap Circuit
fm = ZFeatureMap(feature_dimension=feature_dim)
overlap_circ = UnitaryOverlap(fm.assign_parameters(train_data[0]), fm.assign_parameters(train_data[1]))
overlap_circ.measure_all()

# Visualize the decomposed circuit
overlap_circ.decompose().draw("mpl", scale=.75)
plt.show()

# Run Simulation with Sampler
num_shots = 1000
backend = AerSimulator()
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
optimized_circ = pm.run(overlap_circ)
results = Sampler(mode=backend).run([optimized_circ], shots=num_shots).result()

# Get counts (bitstring & integer labels)
counts_bit = results[0].data.meas.get_counts()
counts_int = results[0].data.meas.get_int_counts()

# Probability of measuring
p_zero = counts_int.get(0, 0.0) / num_shots
print(f"Overlap probability: {p_zero:.3f}")

# Visualization
plot_distribution(counts_bit)
plt.gcf().set_size_inches(5, 5)
plt.tight_layout()
plt.show()