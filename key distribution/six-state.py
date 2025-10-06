from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np

# Utility Functions
def prepare_qubits(qc, bits, bases):
    for n, (bit, base) in enumerate(zip(bits, bases)):
        if base == 0:
            if bit == 1:
                qc.x(n)
        elif base == 1: # X basis
            if bit == 1:
                qc.x(n)
            qc.h(n)
        elif base == 2: # Y basis
            if bit == 1:
                qc.x(n)
            qc.sdg(n)
            qc.h(n)
    return qc

def measure_qubits(qc, bases):
    for n, base in enumerate(bases):
        if base == 1:
            qc.h(n)
        elif base == 2:
            qc.h(n)
            qc.s(n)
        qc.measure(n, n)
    return qc

def simulate_circuit(qc, backend):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc_isa = pm.run(qc)
    job = Sampler(mode=backend).run([qc_isa], shots=1)
    result = job.result()[0].data.c
    key = list(result.get_counts().keys())[0]
    return np.array([int(b) for b in key[::-1]])

def calculate_fidelity(abits, abase, bbits, bbase):
    mask = abase == bbase
    agood = abits[mask]
    bgood = bbits[mask]
    if len(agood) == 0:
        return 0.0, 1.0, np.array([]), np.array([])
    fidelity = np.mean(agood == bgood)
    loss = 1 - fidelity
    return fidelity, loss, agood, bgood

# Six-State Protocol Simulation
def simulate_six_state(bit_num=20, eavesdrop=False, seed=None):
    rng = np.random.default_rng(seed)
    backend = AerSimulator()

    # Alice random bits and bases (0=Z, 1=X, 2=Y)
    abits = rng.integers(0, 2, bit_num)
    abase = rng.integers(0, 3, bit_num)

    # Alice prepares
    qc = QuantumCircuit(bit_num, bit_num)
    qc = prepare_qubits(qc, abits, abase)
    qc.barrier()

    # Optional Eve intercept-resend attack
    if eavesdrop:
        ebase = rng.integers(0, 3, bit_num)
        qc_eve = qc.copy()
        qc_eve = measure_qubits(qc_eve, ebase)
        ebits = simulate_circuit(qc_eve, backend)

        qc = QuantumCircuit(bit_num, bit_num)
        qc = prepare_qubits(qc, ebits, ebase)
        qc.barrier()
    else:
        ebase = None
        ebits = None

    # Bob random bases (0=Z, 1=X, 2=Y)
    bbase = rng.integers(0, 3, bit_num)
    qc = measure_qubits(qc, bbase)
    bbits = simulate_circuit(qc, backend)

    # Calculate fidelity
    fidelity, loss, agood, bgood = calculate_fidelity(abits, abase, bbits, bbase)

    # Display
    print(f"Alice's bits : {abits}")
    print(f"Alice's bases: {abase}")
    if eavesdrop:
        print(f"Eve's bases  : {ebase}")
        print(f"Eve's bits   : {ebits}")
    print(f"Bob's bases  : {bbase}")
    print(f"Bob's bits   : {bbits}")
    print(f"\nMatched bits (Alice): {agood}")
    print(f"Matched bits (Bob)  : {bgood}")
    print(f"Fidelity = {fidelity:.2f}, Loss = {loss:.2f}")

    return {
        "abits": abits, "abase": abase,
        "ebits": ebits, "ebase": ebase,
        "bbits": bbits, "bbase": bbase,
        "fidelity": fidelity, "loss": loss,
        "agood": agood, "bgood": bgood
    }

# Example Usage
print("Without eavesdropper:")
res1 = simulate_six_state(bit_num=24, eavesdrop=False)

print("\nWith eavesdropper:")
res2 = simulate_six_state(bit_num=24, eavesdrop=True)