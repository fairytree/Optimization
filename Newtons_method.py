import sympy as sp

def newtons_method(function, variables, initial_guess, tolerance=1e-6, max_iterations=100, print_vars = True):
    """
    Newton's method for finding extrema of multivariable polynomials.

    Parameters:
        function (sympy.Expr): Multivariable polynomial function.
        variables (list): List of variables in the function.
        initial_guess (list): Initial guess for the extremum.
        tolerance (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: Extremum coordinates.
    """
    # x = sp.symbols(variables)
    gradient = [sp.diff(function, var) for var in variables]
    hessian = [[sp.diff(g, var2) for var2 in variables] for g in gradient]
    x_values = initial_guess
    path = [initial_guess] 
    
    if print_vars:
        print("gradient: ", gradient)       
        print("hessian: ", hessian)
        
    for _ in range(max_iterations):
        gradient_values = sp.Matrix([g.subs(zip(variables, x_values)) for g in gradient])
        hessian_values = sp.Matrix([[h.subs(zip(variables, x_values)) for h in row] for row in hessian])
        hessian_inv = sp.Matrix(hessian_values).inv()    
        delta_x = -hessian_inv * gradient_values   
        x_values = delta_x + x_values
        path.append(x_values)      
        
        if all(abs(d) < tolerance for d in delta_x):
            return x_values, path

    raise ValueError("Method did not converge.")