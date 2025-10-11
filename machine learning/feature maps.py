
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import efficient_su2, z_feature_map, zz_feature_map, pauli_feature_map
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Efficient SU2 Feature Map
x_params = np.linspace(.1, 1.6, 16)
qc_su2 = efficient_su2(num_qubits=4, reps=1, insert_barriers=True)
qc_su2 = qc_su2.assign_parameters(x_params)
qc_su2.decompose().draw("mpl")
plt.show()

# Z Feature Map
z_params = [(1/2) * np.pi / 2, (1/2) * np.pi / 3, (1/2) * np.pi / 4, (1/2) * np.pi / 6]
qc_z = z_feature_map(feature_dimension=4, reps=1)
qc_z = qc_z.assign_parameters(z_params)
qc_z.decompose().draw("mpl")
plt.show()

# ZZ Feature Map
qc_zz = zz_feature_map(feature_dimension=3, entanglement="linear", reps=1)
qc_zz.decompose().draw("mpl")
plt.show()

# Pauli Feature Map
qc_pauli = pauli_feature_map(feature_dimension=3, entanglement="linear", reps=1, paulis=["Y", "XX"])
qc_pauli.decompose().draw("mpl")
plt.show()