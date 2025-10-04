from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer.noise import QuantumError, ReadoutError, pauli_error

# Construct a 1-qubit bit-flip error
p_error = .05
bit_flip = pauli_error([("X", p_error), ("I", 1 - p_error)])
print(bit_flip)

# Construct a 1-qubit phase-flip error
p_error = .05
phase_flip = pauli_error([("Z", p_error), ("I", 1 - p_error)])
print(phase_flip)

# Compose two bit-flip and phase-flip errors
print(bit_flip.compose(phase_flip))

# Tensor product two bit-flip and phase-flip errors with bit-flip on qubit-0, phase-flip on qubit-1
print(phase_flip.tensor(bit_flip))

# Convert to Kraus operator
bit_flip_kraus = Kraus(bit_flip)
print(bit_flip_kraus)

# Convert to Superoperator
phase_flip_sop = SuperOp(phase_flip)
print(phase_flip_sop)

# Convert back to a quantum error
print(QuantumError(bit_flip_kraus))
 
# Check conversion is equivalent to original error
print(QuantumError(bit_flip_kraus) == bit_flip)

# Measurement misassignment probabilities
p0given1, p1given0 = .1 ,.05
print(ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]]))