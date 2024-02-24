import sympy as sp
import numpy as np

def lagrange_multiplier_method(objective_func, constraints, variables):
    # Define Lagrange multiplier symbol
    lambda_ = sp.symbols('lambda_')

    # Define Lagrangian function
    lagrangian = objective_func + lambda_ * constraints

    # Compute gradient of the Lagrangian with respect to variables and lambda
    gradient = [sp.diff(lagrangian, var) for var in variables] + [sp.diff(lagrangian, lambda_)]
    print("gradient: ", sp.latex(gradient))

    # Solve the system of equations
    solution = sp.solve(gradient, variables + [lambda_], dict=True)

    print("solution: ", solution)

    # Extract extremum and Lagrange multiplier from the solution
    extremum =[]
    lagrange_multiplier = []
    for sol in solution:
        extremum.append([sol[var] for var in variables])
        lagrange_multiplier.append(sol[lambda_])

    return extremum, lagrange_multiplier


# question 2a
def function_2a(x):
    return x[0]**2 + x[1]**2 + 10 * x[0] + 20 * x[1] + 25
def constraint_2a(x):
    return x[0] + x[1]

# question 2b
def function_2b(x):
    return - np.pi * x[0]**2 * x[1]
def constraint_2b(x):
    return 2 * np.pi * x[0]**2 + 2 * np.pi * x[0] * x[1] - 24 * np.pi

# question 3
def function_3(x):
    return - np.pi * x[0]**2 * x[1]
def constraint_3(x):
    return 2 * np.pi * x[0]**2 + 2 * np.pi * x[0] * x[1] - 24 * np.pi

# Example usage:
if __name__ == "__main__":
    # Define variables
    x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8')

    # # question 2a
    # variables = [x1, x2]
    # objective_func = function_2a
    # constraint = constraint_2a

    # # question 2b
    # variables = [x1, x2]
    # objective_func = function_2b
    # constraint = constraint_2b

    # question 3
    variables = [x1, x2]
    objective_func = function_3
    constraint = constraint_3

    extremum, lagrange_multiplier = lagrange_multiplier_method(objective_func(variables), constraint(variables), variables)
    for i in range(len(extremum)):
        print("Extremum found at:", extremum[i])
        print("Lagrange multiplier:", lagrange_multiplier[i])    
        print("objective function value:", objective_func(extremum[i]))
