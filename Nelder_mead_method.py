import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2  # Example objective function

def penalty_function(x):
    penalties = 0
    # Example inequality constraint: x[0] + x[1] - 1 <= 0 (penalize if violated)
    if x[0] + x[1] - 1 > 0:
        penalties += 1000 * (x[0] + x[1] - 1)**2

    # Example equality constraint: x[0] - 2 = 0 (treated as inequality for practical purposes)
    epsilon = 0.01  # Small tolerance for equality
    if abs(x[0] - 2) > epsilon:
        penalties += 1000 * (abs(x[0] - 2) - epsilon)**2
    
    return penalties

def modified_objective(x):
    return objective_function(x) + penalty_function(x)

def nelder_mead(func, initial_simplex, tol=1e-5, max_iter=500):
    num_vertices = len(initial_simplex)
    rho = 1
    chi = 2
    gamma = 0.5
    sigma = 0.5

    simplex = initial_simplex

    for i in range(max_iter):
        simplex = sorted(simplex, key=lambda x: func(x))
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + rho * (centroid - simplex[-1])
        if func(simplex[0]) <= func(xr) < func(simplex[-2]):
            simplex[-1] = xr
        elif func(xr) < func(simplex[0]):
            # Expansion
            xe = centroid + chi * (xr - centroid)
            simplex[-1] = xe if func(xe) < func(xr) else xr
        else:
            # Contraction
            xc = centroid + gamma * (simplex[-1] - centroid)
            if func(xc) < func(simplex[-1]):
                simplex[-1] = xc
            else:
                # Shrink
                simplex = [simplex[0]] + [simplex[0] + sigma * (x - simplex[0]) for x in simplex[1:]]

        if np.all(np.abs(func(simplex[0]) - func(simplex)) < tol):
            break

    return simplex[0], func(simplex[0])

# Initial simplex - 3 vertices for a 2D problem
initial_simplex = [np.array([0.5, 0.5]), np.array([0.5, 0]), np.array([0, 0.5])]

opt_solution, opt_value = nelder_mead(modified_objective, initial_simplex)

print("Optimal solution:", opt_solution)
print("Objective function value at optimal solution:", opt_value)
