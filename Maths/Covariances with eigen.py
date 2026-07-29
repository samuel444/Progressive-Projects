import numpy as np

# Example covariance matrix
cov_matrix = np.array([
    [4, 2],
    [2, 3]
])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort largest eigenvalue first
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Explained variance ratios
explained_variance = eigenvalues / eigenvalues.sum()

print("Covariance Matrix:")
print(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

print("\nExplained Variance:")
print(explained_variance)