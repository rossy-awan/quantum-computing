from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

# Signal
n = 10
N = 2**n
x = np.arange(N)
amps = np.random.rand(10)
amps = amps / np.linalg.norm(amps)
freq = [np.random.randint(0, N) for _ in range(10)]
signal = np.sum([amp * np.exp(-1j * 2 * np.pi * f * x / N) for amp, f in zip(amps, freq)], axis=0)

# Quantum state
psi = Statevector(signal / np.linalg.norm(signal))

# QFT
qft = QFT(n)
psi_freq = psi.evolve(qft)

# Frequency distribution
for i in np.argsort(freq):
    print(f'Frequency: {freq[i]}, Amplitude: {round(amps[i], 3)}')
probs = np.abs(psi_freq.data)**2

# Visualization
plt.bar(range(N), probs)
plt.xlabel("Freq")
plt.ylabel("Prob")
plt.tight_layout()
plt.show()