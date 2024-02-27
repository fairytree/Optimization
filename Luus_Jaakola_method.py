import numpy as np

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2  # Example function

# Define penalty for constraints
def penalty(x):
    penalties = 0
    # Example inequality constraint: x[0] + x[1] - 1 <= 0
    if x[0] + x[1] - 1 > 0:
        penalties += (x[0] + x[1] - 1)**2  # Penalty for violating the constraint
    
    # Example equality constraint (transformed into two inequalities): x[0] - 2 = 0
    # Transform equality to a small range around the desired value to act like an inequality for practical purposes
    epsilon = 0.01  # Tolerance for equality
    if abs(x[0] - 2) > epsilon:
        penalties += (x[0] - 2)**2  # Penalty for violating the constraint
    
    return penalties

# Luus-Jaakola method
def luus_jaakola_optimize(func, bounds, constraints_penalty, iterations=1000, search_space_reduction_factor=0.9):
    dim = len(bounds)
    x_best = np.random.rand(dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    f_best = func(x_best) + constraints_penalty(x_best)
    
    for iteration in range(iterations):
        x_new = x_best + (np.random.rand(dim) - 0.5) * 2 * (bounds[:, 1] - bounds[:, 0])
        f_new = func(x_new) + constraints_penalty(x_new)
        
        if f_new < f_best:
            x_best, f_best = x_new, f_new
            bounds = np.vstack((x_best - (bounds[:, 1] - bounds[:, 0]) * search_space_reduction_factor / 2,
                                x_best + (bounds[:, 1] - bounds[:, 0]) * search_space_reduction_factor / 2)).T
            
    return x_best, f_best

# Example: Optimize with constraints
bounds = np.array([[-5, 5], [-5, 5]])  # Define bounds for each variable
result, f_result = luus_jaakola_optimize(objective_function, bounds, penalty)
print("Optimal solution:", result)
print("Objective function value at optimal solution:", f_result)
