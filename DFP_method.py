import numpy as np
import sympy as sp

# DFP optimization function
def dfp_method(func, vars, initial_point, max_iterations=1000, tolerance=1e-5):
    path = [initial_point] 
    x = initial_point
    gradient = [sp.diff(func, var) for var in vars]
    gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient]) # output is 1D array
    H = np.identity(len(initial_point))  # Initial approximation of the inverse Hessian matrix

    iteration = 0
    for _ in range(max_iterations):        
        # Compute the search direction
        p = -np.dot(H, gradient_current) # for 1D array the output is the same as p = -np.dot(gradient_current, H) because np.dot will auto-transpose the 1D array to make the multiplication compatible
        
        # Perform line search to find optimal step size
        alpha = 0.1  # initial step size       
        while func.subs(zip(vars, (x + alpha * p))) > func.subs(zip(vars, x)) + 0.5 * alpha * np.dot(gradient_current, p):
            alpha *= 0.5
            
        x_new = x + alpha * p  # Update the current point
        
        grad_x_new = np.array([float(g.subs(zip(vars, x_new))) for g in gradient])
        delta_g = np.array(grad_x_new - gradient_current)
        
        # The delta_x is a 1D array, transpose has no effect on 1D array. It has to be re-shaped as a column vector
        delta_x = np.array(x_new - x).reshape((-1, 1))
        
        # Update the inverse Hessian matrix using the DFP formula
        H = H + np.outer(delta_x, delta_x) / np.dot(delta_x.T, delta_g) - np.dot(np.dot(H, np.outer(delta_g, delta_g)), H) / np.dot(delta_g.T, np.dot(H, delta_g))
       
        x = x_new
        path.append(x)
        gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient])

        if np.linalg.norm(gradient_current) < tolerance:
            return x, path

    raise ValueError("Method did not converge.")
