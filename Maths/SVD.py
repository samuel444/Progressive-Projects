import numpy as np

A = np.array([
    [3, 1],
    [1, 3],
    [1, 1]
])

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("U:")
print(U)

print("\nSingular values:")
print(S)

print("\nV^T:")
print(Vt)

# Reconstruct original matrix
A_reconstructed = U @ np.diag(S) @ Vt

print("\nReconstructed A:")
print(A_reconstructed)