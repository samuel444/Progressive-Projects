import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Example data: rows = observations, columns = features
X = np.array([
    [2, 1],
    [3, 2],
    [4, 2],
    [5, 4],
    [6, 5]
])

# Standardise data
X_scaled = StandardScaler().fit_transform(X)


# PCA manually using covariance + eigenvectors
cov_matrix = np.cov(X_scaled, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort largest eigenvalue first
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Explained variance ratios
explained_variance_ratio = (
    eigenvalues / eigenvalues.sum()
)

# Transform original data into principal components
X_pca_manual = X_scaled @ eigenvectors

print("Eigenvalues:")
print(eigenvalues)

print("\nExplained variance ratios:")
print(explained_variance_ratio)

print("\nEigenvectors / PC coefficients:")
print(eigenvectors)

print("\nManual PCA data:")
print(X_pca_manual)


# Compare with sklearn PCA
pca = PCA()
X_pca_sklearn = pca.fit_transform(X_scaled)

print("\nSklearn explained variance ratios:")
print(pca.explained_variance_ratio_)

print("\nSklearn PC coefficients:")
print(pca.components_)

print("\nSklearn PCA data:")
print(X_pca_sklearn)