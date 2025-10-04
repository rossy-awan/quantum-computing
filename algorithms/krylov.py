import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Operator
plt.rcParams.update({'font.family': 'Cambria', 'font.style': 'italic', 'mathtext.fontset': 'cm', 'font.size': 16})

# Utility Functions
def orthog_pair(v_known: np.ndarray, v_next: np.ndarray) -> np.ndarray:
    v_known = v_known / np.sqrt(v_known.T @ v_known)
    return v_next - (v_known.T @ v_next) * v_known

def orthoset(v: np.ndarray, subspace: list[np.ndarray]) -> np.ndarray:
    v = v / np.sqrt(v.T @ v)
    temp = v.copy()
    for s in subspace:
        temp = orthog_pair(s, temp)
    return temp / np.sqrt(temp.T @ temp)

def errors(matrix: np.ndarray, krylov_eigs: list[np.ndarray]) -> list[float]:
    return [np.min(eigs_i) - np.min(np.linalg.eigvals(matrix)) for eigs_i in krylov_eigs]

# Krylov Subspace Builder
def krylov_full_build(v0: np.ndarray, matrix: np.ndarray, tolerance: float = 1e-8):
    A = np.array(matrix, dtype=float)
    b = v0 / np.sqrt(v0 @ v0.T)
    ks = [b]
    Hs = [np.array([[b.T @ A @ b]])]
    eigs = [np.array([b.T @ A @ b])]
    for j in range(len(A) - 1):
        # Expand Krylov basis
        vec = A @ ks[j].T
        ks.append(orthoset(vec, ks))

        # Project Hamiltonian to subspace
        ks_mat = np.array(ks)
        H_proj = ks_mat @ A @ ks_mat.T
        Hs.append(H_proj)

        # Diagonalize the projected Hamiltonian
        eigenvals = np.linalg.eig(H_proj).eigenvalues
        eigs.append(eigenvals)
        if abs(np.min(eigs[-2]) - np.min(eigenvals)) < tolerance:
            break
        
    return ks, Hs, eigs

#  Hamiltonian
num_spins, ham_list = 8, []
for i in range(num_spins):
    j = (i + 1) % num_spins # periodic connection
    ham_list.append(("XX", [i, j], .5))
    ham_list.append(("ZZ", [i, j], .5))
for i in range(num_spins):
    ham_list.append(("X", [i], np.random.random() * 2 - 1))
    ham_list.append(("Z", [i], np.random.random() * 2 - 1))
hamiltonian = SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_spins)
H = np.array(np.real(Operator(hamiltonian).data), dtype=float)
ks, Hs, eigs = krylov_full_build(np.random.rand(2 ** num_spins), H)

# Compare with true eigenvalues
print(f"True eigenvalues: {np.sort(np.linalg.eigvals(H))[:10]}")
print(f"\nKrylov final approximation: {np.sort(eigs[-1])[:10]}")

# Plot Results
plt.figure(figsize=(5, 5))
plt.plot(errors(H, eigs), color='k', lw=1)
plt.axhline(y=0, color="red", lw=1)
plt.xlabel(r"$r$")
plt.ylabel(r"$\epsilon$")
plt.tight_layout()
plt.show()