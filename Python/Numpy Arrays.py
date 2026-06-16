import numpy as np

# Create arrays 
arr = np.array([10, 20, 30, 40, 50])

# Indexing and Slicing
print(arr[0])      # First element
print(arr[2])      # Third element
print(arr[-1])     # Last element

print(arr[1:4])    # Slice
print(arr[:3])     # First 3 elements
print(arr[2:])     # From index 2 onwards


print()

# Arithmetic Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + 5)       # Add scalar
print(a + b)       # Addition
print(a - b)       # Subtraction
print(a * b)       # Element-wise multiplication
print(b / a)       # Division


print()

# Aggregation Operations
print(np.sum(a))
print(np.mean(a))
print(np.max(a))
print(np.min(a))
print(np.std(a)) # Standard deviation


print()

# Matrix Operations
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(A @ B)                # Matrix multiplication
print(A.T)                  # Transpose
print(np.linalg.det(A))     # Determinant
print(np.linalg.inv(A))     # Inverse