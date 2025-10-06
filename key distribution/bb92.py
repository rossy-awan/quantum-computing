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
def simulate_bb92(bit_num=20, seed=None):
    rng = np.random.default_rng(seed)
    backend = AerSimulator()

    # Alice random bits
    abits = rng.integers(0, 2, bit_num)

    # Bob's random measurement bases (Z=0, X=1)
    bbase = rng.integers(0, 2, bit_num)

    # Prepare Alice's signals: |0> for 0, |+> for 1
    qc = QuantumCircuit(bit_num, bit_num)
    for n, abit in enumerate(abits):
        if abit == 1:
            qc.h(n) # prepares |+> from |0>
        # if abit == 0: do nothing (|0>)
    qc.barrier()

    # Bob's measurement: apply H when measuring in X basis, then measure
    for n, bb in enumerate(bbase):
        if bb == 1:
            qc.h(n)
        qc.measure(n, n)

    # Run circuit and get Bob's raw bits
    braw = simulate_circuit(qc, backend)

    # Post-selection / inference for BB92:
    # - If Bob measured Z (bbase==0) and observed 1 -> conclusive -> infer bit 1
    # - If Bob measured X (bbase==1) and observed 1 -> conclusive -> infer bit 0
    inferred_positions = []
    inferred_bits = []
    for n in range(bit_num):
        if bbase[n] == 0 and braw[n] == 1:
            inferred_positions.append(n)
            inferred_bits.append(1)
        elif bbase[n] == 1 and braw[n] == 1:
            inferred_positions.append(n)
            inferred_bits.append(0)
        # else inconclusive -> skip

    inferred_positions = np.array(inferred_positions, dtype=int)
    inferred_bits = np.array(inferred_bits, dtype=int)

    # Alice's bits at those positions
    if len(inferred_positions) > 0:
        alice_at_inferred = abits[inferred_positions]
        fidelity = np.mean(alice_at_inferred == inferred_bits)
        loss = 1 - fidelity
    else:
        alice_at_inferred = np.array([], dtype=int)
        fidelity = 0.0
        loss = 1.0

    # Print summary
    print(f"Alice bits         : {abits}")
    print(f"Bob bases (Z=0,X=1): {bbase}")
    print(f"Bob raw results    : {braw}")
    print(f"Inferred positions : {inferred_positions}")
    print(f"Inferred bits      : {inferred_bits}")
    print(f"Alice @ inferred   : {alice_at_inferred}")
    print(f"Conclusive rounds  : {len(inferred_bits)} / {bit_num}")
    print(f"Fidelity (on conclusive subset) = {fidelity:.3f}")
    print(f"Loss = {loss:.3f}")
    
    return {
        "abits": abits,
        "bbase": bbase,
        "braw": braw,
        "inferred_positions": inferred_positions,
        "inferred_bits": inferred_bits,
        "alice_at_inferred": alice_at_inferred,
        "fidelity": fidelity,
        "loss": loss
    }

# Example usage:
res = simulate_bb92(bit_num=25)