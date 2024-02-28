import numpy as np


def objective(x):
    return np.sum(x**2)

# inequality constraints
def update_x(z, u, rho):
    return np.clip(z - u, 0, None)  # Ensures x is non-negative

# equality constraints
def update_z(x, u, A, b, rho):
    """Updates z based on equality constraints: Ax = b."""
    AtA = A.T @ A
    Atb = A.T @ b
    system_matrix = AtA + rho * np.eye(A.shape[1])
    rhs = Atb + rho * (x + u)
    return np.linalg.solve(system_matrix, rhs)

#  Updates the dual variable u
def update_u(u, x, z):
    return u + x - z

def admm_optimization(bounds, A, b, rho=1.0, max_iter=100, epsilon_pri=1e-4, epsilon_dual=1e-4):
    n_variables = len(bounds)
    x = np.random.rand(n_variables) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    z = np.random.randn(n_variables)
    u = np.zeros_like(x)

    for i in range(max_iter):
        x = update_x(z, u, rho)
        z = update_z(x, u, A, b, rho)
        u = update_u(u, x, z)
        
        primal_residual = np.linalg.norm(x - z)
        dual_residual = np.linalg.norm(rho * (z - np.copy(z)))

        if primal_residual < epsilon_pri and dual_residual < epsilon_dual:
            print(f"Converged at iteration {i + 1}")
            break

    return x, objective(x)

# Define problem parameters, Ax = b is the equal constraints
A = np.array([[1, 2], [3, 4]])  
b = np.array([1, 1])           
bounds = np.array([[-10, 10], [-10, 10]])  

opt_solution, opt_value = admm_optimization(bounds, A, b)

print("Optimal solution:", opt_solution)
print("Objective value:", opt_value)
