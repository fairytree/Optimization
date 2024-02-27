import numpy as np

def pivot(N, B, A, b, c, v, l, e):
    # Let the entering variable be x_e, leaving variable be x_l
    # Update the coefficients and the objective value
    b[l] = b[l] / A[l][e]
    A[l] = A[l] / A[l][e]
    for i in B:
        if i != l:
            b[i] = b[i] - A[i][e] * b[l]
            A[i] = A[i] - A[i][e] * A[l]
    v = v + c[e] * b[l]
    c = c - c[e] * A[l]
    # Update basis and non-basis
    N[N == e] = l
    B[B == l] = e
    return N, B, A, b, c, v

def simplex(c, A, b):
    # Initialize
    m, n = A.shape
    v = 0
    N = np.arange(n, n + m)
    B = np.arange(n)
    # Run Simplex algorithm
    while True:
        e = -1
        for j in N:
            if c[j] > 0:
                e = j
                break
        if e == -1: break  # Optimal reached
        l = -1
        for i in B:
            if A[i][e] > 0:
                if l == -1 or b[i] / A[i][e] < b[l] / A[l][e]:
                    l = i
        if l == -1: raise Exception("Unbounded")
        N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)
    # Construct solution
    x = np.zeros(n + m)
    for i in B:
        if i < n: x[i] = b[i]
    return x[:n], v

# Example
c = np.array([5, 4, 0, 0])  # Objective function coefficients
A = np.array([[6, 4, 1, 0], [1, 2, 0, 1]])  # Constraints coefficients
b = np.array([24, 6])  # Constraints bounds

x, v = simplex(c, A, b)
print("Optimal solution:", x)
print("Maximum value:", v)
