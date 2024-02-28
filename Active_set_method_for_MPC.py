import numpy as np

def objective(x):
    """Objective function to be minimized."""
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def eq_constraint(x):
    """Equality constraint: x[0] + 2*x[1] = 2"""
    return x[0] + 2 * x[1] - 2

def ineq_constraint(x):
    """Inequality constraint: x[0] + x[1] <= 5"""
    return 5 - (x[0] + x[1])

def check_constraints(x, active_set):
    """Check which constraints are active/violated."""
    is_active = []
    for constraint in active_set:
        if constraint['type'] == 'eq' or (constraint['type'] == 'ineq' and constraint['fun'](x) < 0):
            is_active.append(True)
        else:
            is_active.append(False)
    return is_active

def active_set_optimizer(x0, eq_cons, ineq_cons, tol=1e-5, max_iter=100):
    """A simplified Active Set optimizer."""
    x = x0
    active_set = eq_cons + ineq_cons  # Start with all constraints considered active

    for iteration in range(max_iter):
        # Simplification: Assume linear or quadratic problem solved here for x
        # This step would involve solving a QP problem considering the active set
        
        # Placeholder for x update (should be solved based on active constraints)
        # In a full implementation, this would involve solving a QP subproblem
        x = np.random.rand(2)  # Placeholder: Randomly update x (replace with actual optimization step)
        
        # Check active constraints
        is_active = check_constraints(x, active_set)
        
        # Update active set based on constraint check
        # This is a placeholder. Actual implementation would add/remove constraints based on violations and optimality
        if all(is_active):
            print(f"Convergence achieved at iteration {iteration}")
            break
    
    return x

# Define constraints
eq_cons = [{'type': 'eq', 'fun': eq_constraint}]
ineq_cons = [{'type': 'ineq', 'fun': ineq_constraint}]

# Initial guess
x0 = np.array([0.5, 0.5])

# Optimize
opt_solution = active_set_optimizer(x0, eq_cons, ineq_cons)
print("Optimal solution:", opt_solution)
print("Value of Function at extremum: ", objective(opt_solution))
