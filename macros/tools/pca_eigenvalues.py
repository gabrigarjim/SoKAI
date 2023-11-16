import numpy as np

matrix = []

file_path="correlation_matrix.txt"

with open(file_path, 'r') as file:
 for line in file:
  row = [ float(x) for x in line.split()]
  matrix.append(row)


print("Matrix : ", matrix)
eigenvalues, eigenvectors = np.linalg.eig(matrix)

weights = [i / sum(eigenvalues) for i in eigenvalues]
print("Weights: " , weights)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
