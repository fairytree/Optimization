import numpy as np
import sympy as sp

def fletcher_reeves(f, vars, initial_point, max_iterations=1000, tolerance=1e-5):
    # Step 1: At X0 calculate F(x0). Let p0 = -gradient of F(x0)
    x = initial_point
    gradient = [sp.diff(f, var) for var in vars]
    gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient])
    p = -gradient_current

    # Test for convergence and terminate the algorithm when norm of p(k) < TOL (prescribed tolerance)
    iteration = 0
    while np.linalg.norm(gradient_current) > tolerance and iteration < max_iterations:
        # Perform line search to find optimal step size
        alpha = 0.01  # initial step size
        while f.subs(zip(vars, (x + alpha * p))) > f.subs(zip(vars, x)) + 0.5 * alpha * np.dot(gradient_current, p):
            alpha *= 0.5
        
        # Step 2: Save gradient and compute x(k+1) = x(k) + alpha(k) * p(k)
        gradient_previous = gradient_current
        x = x + alpha * p

        # Step 3: Calculate F(x(k+1)), gradient of F(x(k+1)). 
        gradient = [sp.diff(f, var) for var in vars]
        gradient_current = np.array([float(g.subs(zip(vars, x))) for g in gradient])
        beta = np.dot(gradient_current, gradient_current) / np.dot(gradient_previous, gradient_previous)
        p = -gradient_current + beta * p
        iteration += 1
    return x


def polynomial(x):
    # Define your multivariable polynomial here
    # Example: f(x, y) = x^2 + y^2
    return 4*(x[0] - 2)**2 + (x[1] - 6)**2

if __name__ == "__main__":
    x, y = sp.symbols('x y')
    vars = [x, y]
    initial_point = np.array([1.0, 1.0])  # Initial point
    extremum = fletcher_reeves(polynomial(vars), vars, initial_point)
    print("Extremum found at:", extremum)
    print("Value of polynomial at extremum:", polynomial(extremum))
