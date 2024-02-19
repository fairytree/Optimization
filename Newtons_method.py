import sympy as sp

def newtons_method(f, variables, initial_guess, tolerance=1e-6, max_iterations=100):
    """
    Newton's method for finding extrema of multivariable polynomials.

    Parameters:
        f (sympy.Expr): Multivariable polynomial function.
        variables (list): List of variables in the function.
        initial_guess (list): Initial guess for the extremum.
        tolerance (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: Extremum coordinates.
    """
    x = sp.symbols(variables)
    gradient = [sp.diff(f, var) for var in x]
    print("gradient: ", gradient)
    hessian = [[sp.diff(g, var2) for var2 in x] for g in gradient]
    print("hessian: ", hessian)

    x_values = initial_guess
    print("x_values: ", x_values)
    for _ in range(max_iterations):
        gradient_values = sp.Matrix([g.subs(zip(x, x_values)) for g in gradient])
        print("gradient_values: ", gradient_values)
        
        hessian_values = sp.Matrix([[h.subs(zip(x, x_values)) for h in row] for row in hessian])
        print("hessian_values: ", hessian_values)
        
        hessian_inv = sp.Matrix(hessian_values).inv()
        print("hessian_inv: ", hessian_inv)
              
        delta_x = -hessian_inv * gradient_values
        print("delta_x: ", delta_x)
        
        x_values = delta_x + x_values
        print("x_values: ", x_values)
        
        if all(abs(d) < tolerance for d in delta_x):
            return x_values

    raise ValueError("Newton's method did not converge.")


if __name__ == "__main__":
    # Example usage
    x, y = sp.symbols('x y')
    f = x**2 + y**2  # Example polynomial function, change as needed
    variables = ['x', 'y']
    initial_guess = sp.Matrix([1, 1])  # Initial guess for the extremum
    extremum = newtons_method(f, variables, initial_guess)
    print("Extremum found at:", extremum)
