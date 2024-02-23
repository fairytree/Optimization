import numpy as np
import sympy as sp

def fletcher_reeves(f, vars, initial_point, max_iterations=1000, tolerance=1e-5, print_vars = True):
    # Step 1: At X0 calculate F(x0). Let p0 = -gradient of F(x0)
    path = [initial_point] 
    x = initial_point
    gradient = [sp.diff(f, var) for var in vars]
    gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient])
    p = -gradient_current

    if print_vars:
        print("gradient: ", gradient)       
        
    # Test for convergence and terminate the algorithm when norm of p(k) < TOL (prescribed tolerance)  
    for _ in range(max_iterations):
        # Perform line search to find optimal step size
        alpha = 0.1  # initial step size
        while f.subs(zip(vars, (x + alpha * p))) > f.subs(zip(vars, x)) + 0.5 * alpha * np.dot(gradient_current, p):
            alpha *= 0.5
        
        # Step 2: Save gradient and compute x(k+1) = x(k) + alpha(k) * p(k)
        gradient_previous = gradient_current
        x = x + alpha * p
        path.append(x) 

        # Step 3: Calculate F(x(k+1)), gradient of F(x(k+1)). 
        gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient])
        beta = np.dot(gradient_current, gradient_current) / np.dot(gradient_previous, gradient_previous)
        p = -gradient_current + beta * p
        
        if np.linalg.norm(p) < tolerance:
            return x, path

    raise ValueError("Method did not converge.")

