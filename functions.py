import sympy as sp
import visualization as vis
import numpy as np
import Newtons_method as newton
import Fletcher_Reeves_method as FRM
import DFP_method as DFP

# question 1a-1 Newton's method
def function_1a1(x):
    return 1 + x[0] + x[1] + x[2] + x[3] + x[0] * x[1] + x[0] * x[2] + x[0] * x[3] + x[1] * x[2] + x[1] * x[3] + x[2] * x[3] + x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2


# question 1a-2 Newton's method
def function_1a2(x):
    return 8 * x[0]**2 + 4 * x[0] * x[1] + 5 * x[1]**2


# question 1b Fletcher_Reeves Method 
def function_1b(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2


# question 1c DFP method
def function_1c(x):
    return x[0] - x[1] + 2 * x[0]**2 + 2 * x[0] * x[1] + x[1]**2


if __name__ == "__main__":
    x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8')
    
    # #-----question 1a-1 Newton's Method with initial_guess = sp.Matrix([-3.0, -30, -4, -0.1])-----
    # variables = [x1, x2, x3, x4]
    # initial_guess = sp.Matrix([-3.0, -30, -4, -0.1])
    # function = function_1a1
    # extremum, path  = newton.newtons_method(function(variables), variables, initial_guess)
    
    # #-----question 1a-1 Newton's Method with initial_guess = sp.Matrix([0.5, 1.0, 8.0, -0.7])--------
    # variables = [x1, x2, x3, x4]
    # initial_guess = sp.Matrix([0.5, 1.0, 8.0, -0.7])
    # function = function_1a1
    # extremum, path  = newton.newtons_method(function(variables), variables, initial_guess)
    
    # # ------question 1a-2 Newton's Method ----------
    # variables = [x1, x2]
    # initial_guess = sp.Matrix([10.0, 10.0])
    # function = function_1a2
    # extremum, path  = newton.newtons_method(function(variables), variables, initial_guess)
    
    # # ------question 1b Fletcher_Reeves Method --------
    # variables = [x1, x2]
    # initial_guess = np.array([1.0, 1.0])  # Initial point
    # function = function_1b
    # extremum, path = FRM.fletcher_reeves(function(variables), variables, initial_guess)
    
    # # ------question 1c DFP Method --------
    variables = [x1, x2]
    initial_guess = np.array([0.0, 0.0])  # Initial point, this is a 1D array
    function = function_1c
    extremum, path = DFP.dfp_method(function(variables), variables, initial_guess)
    
    
    vis.data_table(function, extremum, variables, path)
    vis.plot(function, path, variables)
    


    
    
    