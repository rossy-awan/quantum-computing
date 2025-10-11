from qiskit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm'})

class QuantumEncoder:
    @staticmethod
    def basis_encode(data, num_bits=4, show=False):
        data = np.array(data)
        all_bits = []
        if np.any(data < 0):
            raise ValueError("Basis encoding hanya menerima bilangan bulat non-negatif.")
        for val in data:
            bits = [int(b) for b in format(val, f"0{num_bits}b")]
            all_bits.extend(bits)
        num_qubits = len(all_bits)
        qc = QuantumCircuit(num_qubits)
        for i, bit in enumerate(all_bits):
            if bit == 1:
                qc.x(i)
        if show:
            qc.draw("mpl")
            plt.show()
        return qc

    @staticmethod
    def amplitude_encode(data, show=False):
        arr = np.array(data, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Input data must not be all zeros.")
        normalized = arr / norm
        num_qubits = int(np.ceil(np.log2(len(normalized))))
        padded_len = 2**num_qubits
        if len(normalized) < padded_len:
            normalized = np.pad(normalized, (0, padded_len - len(normalized)))
        qc = QuantumCircuit(num_qubits)
        qc.initialize(normalized, qc.qubits)
        if show:
            qc.decompose(reps=5).draw("mpl")
            plt.show()
        return qc

    @staticmethod
    def phase_encode(data, normalize=False, show=False):
        data = np.array(data, dtype=float)
        if normalize:
            data = 2 * np.pi * (data - np.min(data)) / (np.ptp(data) + 1e-12)
        num_qubits = len(data)
        qc = QuantumCircuit(num_qubits, name="PhaseEncode")
        qc.h(range(num_qubits))
        for i, phase_value in enumerate(data):
            qc.p(phase_value, i)
        if show:
            qc.draw("mpl")
            plt.show()
        return qc

# Basis encoding
qc_basis = QuantumEncoder.basis_encode([7, 3, 5], num_bits=3, show=True)

# Amplitude encoding
qc_amp = QuantumEncoder.amplitude_encode([4, 8, 5, 0], show=True)

# Phase encoding
qc_phase = QuantumEncoder.phase_encode([np.pi/4, np.pi/2, np.pi, 3*np.pi/2], show=True)