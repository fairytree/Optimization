import sympy as sp
import visualization as vis
import tabulate as tab

def newtons_method(f, variables, initial_guess, tolerance=1e-6, max_iterations=100, print_vars = False):
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
    hessian = [[sp.diff(g, var2) for var2 in x] for g in gradient]
    x_values = initial_guess
    path = [initial_guess] 
    
    if print_vars:
        print("gradient: ", gradient)       
        print("hessian: ", hessian)
        print("x_values: ", x_values)
        
    for _ in range(max_iterations):
        gradient_values = sp.Matrix([g.subs(zip(x, x_values)) for g in gradient])
        hessian_values = sp.Matrix([[h.subs(zip(x, x_values)) for h in row] for row in hessian])
        hessian_inv = sp.Matrix(hessian_values).inv()    
        delta_x = -hessian_inv * gradient_values   
        x_values = delta_x + x_values
        path.append(x_values)      
        
        if print_vars:
            print("x_values: ", x_values)
            print("delta_x: ", delta_x)
            print("hessian_inv: ", hessian_inv)
            print("hessian_values: ", hessian_values)
            print("gradient_values: ", gradient_values)
        
        if all(abs(d) < tolerance for d in delta_x):
            if print_vars:
                print("path: ", path)
            return x_values, path

    raise ValueError("Newton's method did not converge.")


if __name__ == "__main__":
    # variables
    variables = ["x1", "x2", "x3", "x4"]
    # variables = ["x", "y"]
    
    # this piece of code needs to be executed before the definition of f
    for var in variables:
        globals()[var] = sp.symbols(var) # assigns a variable with the name var the value sp.symbols(var)  
    
    # variables continued
    f = 1 + x1 + x2 + x3 + x4 + x1 * x2 + x1 * x3 + x1 * x4 + x2 * x3 + x2 * x4 + x3 * x4 + x1**2 + x2**2 + x3**2 + x4**2 # ignore the yellow underline
    initial_guess = sp.Matrix([0.5, 1.0, 8.0, -0.7])
    # f = x**2 + y**2
    # initial_guess = sp.Matrix([1, 1])
    visualize = False
    buffer = 1 # this is for visualization
    grid = [100, 100] # this is also for visualization
    print_variables = False
    
    # find extremums
    extremum, path  = newtons_method(f, variables, initial_guess, print_vars=print_variables)
    print("Extremum found at:", extremum)
    subed_vars = {}
    for i in range(len(variables)):
        subed_vars[variables[i]] = extremum[i]
    f_of_extremum = f.evalf(subs=subed_vars)
    print("Extremum: ", f_of_extremum)
    
    # visualize
    if visualize:
        grid = [100, 100]
        path_x = []
        path_y = []
        for pt in path:
            path_x.append(pt[0])
            path_y.append(pt[1])
        x_range = [float(min(path_x)) - buffer, float(max(path_x)) + buffer]
        y_range = [float(min(path_y)) - buffer, float(max(path_y)) + buffer]
        data = []
        for i in range(len(path)):
            data.append([i + 1, path[i][0], path[i][1], f.evalf(subs={x: path[i][0], y: path[i][1]})])
        print(tab.tabulate([[None, "x", "y", "f(x)"]] + data, tablefmt="simple_grid"))
        vis.visualize(f, x_range, y_range, grid, path_x, path_y)
