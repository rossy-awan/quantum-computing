from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np

# Utility Functions
def simulate_circuit(qc, backend):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc_isa = pm.run(qc)
    job = Sampler(mode=backend).run([qc_isa], shots=1)
    result = job.result()[0].data.c
    key = list(result.get_counts().keys())[0]
    return np.array([int(b) for b in key[::-1]]) # reverse endian

# Main Simulation Function
def simulate_bbm92(bit_num=10, seed=None):
    rng = np.random.default_rng(seed)
    backend = AerSimulator()

    # Random measurement bases for Alice and Bob (Z=0, X=1)
    abase = rng.integers(0, 2, bit_num)
    bbase = rng.integers(0, 2, bit_num)

    # Create entangled pairs (EPR states)
    qc = QuantumCircuit(2 * bit_num, 2 * bit_num)
    for n in range(bit_num):
        qc.h(2 * n)
        qc.cx(2 * n, 2 * n + 1)
    qc.barrier()

    # Alice measures in her chosen basis
    for n, base in enumerate(abase):
        if base == 1:
            qc.h(2 * n)
        qc.measure(2 * n, 2 * n)

    # Bob measures in his chosen basis
    for n, base in enumerate(bbase):
        if base == 1:
            qc.h(2 * n + 1)
        qc.measure(2 * n + 1, 2 * n + 1)

    # Simulate
    results = simulate_circuit(qc, backend)

    # Split results into Alice and Bob outcomes
    alice_bits = results[0::2]
    bob_bits = results[1::2]

    # Keep only matched bases
    mask = abase == bbase
    alice_good = alice_bits[mask]
    bob_good = bob_bits[mask]

    if len(alice_good) > 0:
        fidelity = np.mean(alice_good == bob_good)
        loss = 1 - fidelity
    else:
        fidelity = 0
        loss = 1

    # Output summary
    print(f"Alice bases (Z = 0, X = 1) : {abase}")
    print(f"Bob bases   (Z = 0, X = 1) : {bbase}")
    print(f"Alice bits                 : {alice_bits}")
    print(f"Bob bits                   : {bob_bits}")
    print(f"Matched bases indices      : {np.where(mask)[0]}")
    print(f"Alice matched bits         : {alice_good}")
    print(f"Bob matched bits           : {bob_good}")
    print(f"Matched rounds             : {len(alice_good)} / {bit_num}")
    print(f"Fidelity                   : {fidelity:.3f}")
    print(f"Loss                       : {loss:.3f}")

    return {
        "abase": abase,
        "bbase": bbase,
        "alice_bits": alice_bits,
        "bob_bits": bob_bits,
        "alice_good": alice_good,
        "bob_good": bob_good,
        "fidelity": fidelity,
        "loss": loss,
        "matched": mask
    }

# Example usage
res = simulate_bbm92(bit_num=12)