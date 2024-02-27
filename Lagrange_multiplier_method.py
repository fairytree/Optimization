import sympy as sp
import numpy as np
from scipy.optimize import minimize

def lagrange_multiplier_method(objective_func, constraints, variables):
    # Define Lagrange multiplier symbol and Lagrangian function
    lambdas = sp.symbols('lambda_0:%d' % len(constraints)) # if len(constraints) is 3, then lambdas = (lambda_0, lambda_1, lambda_2)
    print("lambdas: ", lambdas)
    lagrangian = objective_func
    for i, constraint in enumerate(constraints):
        lagrangian += lambdas[i] * constraint
    print("lagrangian: ", lagrangian)
    # Compute gradient of the Lagrangian with respect to variables and lambda
    gradient = [sp.diff(lagrangian, var) for var in variables] + [sp.diff(lagrangian, lam) for lam in lambdas]
    print("gradient: ", sp.latex(gradient))

    # Solve the system of equations
    solution = sp.solve(gradient, variables + list(lambdas), dict=True)
    print("solution: ", solution)

    # Extract extremum and Lagrange multipliers from the solution
    extrema = [[sol[var] for var in variables] for sol in solution]
    lagrange_multipliers = [[sol[lam] for lam in lambdas] for sol in solution]

    # Compute Hessian matrix of the Lagrangian function and evaluate Hessian matrix at the extremum point
    hessian_matrix = sp.hessian(lagrangian, variables)
    hessian_value_at_extrema = [hessian_matrix.subs(zip(variables, extr)) for extr in extrema]
    print("hessian_matrix: ", hessian_matrix)
    print("hessian_value_at_extrema: ", hessian_value_at_extrema)

    return extrema, lagrange_multipliers


# question 2a
def function_2a(x):
    return x[0]**2 + x[1]**2 + 10 * x[0] + 20 * x[1] + 25
def constraint_2a(x):
    return [x[0] + x[1]]

# question 2b
def function_2b(x):
    return - np.pi * x[0]**2 * x[1]
def constraint_2b(x):
    return [2 * np.pi * x[0]**2 + 2 * np.pi * x[0] * x[1] - 24 * np.pi]

# question 3
def function_3(x):
    n = 10
    P = 750
    f = 0
    w = [-10.021, -21.096, -37.986, -9.846, -28.653, -18.918, -28.032, -14.640, -30.594, -26.111]
    sum_x = 0
    for j in range(n):
        sum_x += x[j]
    for i in range(n):
        f = f + x[i] * (w[i] + sp.log(P) + sp.log(x[i]/sum_x))
    return f
def constraint_3(x):
    return x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2, x[3] + 2 * x[4] + x[5] + x[6] - 1, x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1
def constraint_3_1(x):
    return x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2
def constraint_3_2(x):
    return x[3] + 2 * x[4] + x[5] + x[6] - 1
def constraint_3_3(x):
    return x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1


# question 4
def function_4(x):
    return (x[0]-2)**2 + x[1]**2 + (x[2]-1)**2
def constraint_4(x):
    return [x[2]**2 - 4 * x[0]**2 - 2 * x[1]**2]


# question 5a
def function_5a(x):
    return x[0]**2 + x[1]**2 - 14 * x[0] - 6 * x[1] -7
def constraint_5a_ineq(x):
    return [x[0] + x[1] - 2, x[0] + 2 * x[1] - 3]

# question 5b
def function_5b(x):
    return x[0]**2 + 2 * (x[1] + 1)**2
def constraint_5b_eq(x):
    return [-x[0] + x[1] - 2]
def constraint_5b_ineq(x):
    return [-x[0] - x[1] - 1]

# question 6a
def function_6a(x):
    return x[0]**2 + 3 / 2 * x[1]**2 - 4 * x[0] - 7 * x[1] + x[0] * x[1] + 9 - sp.log(x[0]) - sp.log(x[1])
def constraint_6a_ineq(x):
    return []

# question 6b
def function_6b(x):
    return x[0]**2 + 3 / 2 * x[1]**2 - 4 * x[0] - 7 * x[1] + x[0] * x[1] + 9 - sp.log(x[0]) - sp.log(x[1])
def constraint_6b_ineq(x):
    return [4 - x[0] * x[1]]

# question 6c
def function_6c(x):
    return x[0]**2 + 3 / 2 * x[1]**2 - 4 * x[0] - 7 * x[1] + x[0] * x[1] + 9 - sp.log(x[0]) - sp.log(x[1]) 
def constraint_6c_eq(x):
    return [2 * x[0] - x[1]]
def constraint_6c_ineq(x):
    return [4 - x[0] * x[1]]


# Example usage:
if __name__ == "__main__":
    # Define variables
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')

    # # question 2a
    # variables = [x1, x2]
    # objective_func = function_2a
    # constraint = constraint_2a(variables)

    # # question 2b
    # variables = [x1, x2]
    # objective_func = function_2b
    # constraint = constraint_2b(variables)

    # # question 3
    # initial_guess = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # objective_func = function_3
    # # constraints = [{'type': 'eq', 'fun': constraint_3_1},
    # #             {'type': 'eq', 'fun': constraint_3_2},
    # #             {'type': 'eq', 'fun': constraint_3_3}]
    # constraints = [{'type': 'eq', 'fun': constraint_3_1},
    #                {'type': 'eq', 'fun': constraint_3_1}]
    # method = 'trust-constr'  # 'SLSQP': no optimization is done; 'COBYLA': type 'eq' not handled by COBYLA
    # bounds = [(0, 1), (0, 1),(0, 1), (0, 1),(0, 1), (0, 1),(0, 1), (0, 1),(0, 1), (0, 1)]
    # result = minimize(objective_func, initial_guess, constraints=constraints, method=method, bounds=bounds)
    # print("Optimal solution:", result.x)
    # print("Optimal value:", result.fun)

    # # question 4
    # variables = [x1, x2, x3]
    # objective_func = function_4
    # constraint = constraint_4(variables)

    # # question 5a
    # variables = [x1, x2]
    # objective_func = function_5a
    # constraint_ineq = constraint_5a_ineq(variables)
    # constraint_eq = None
    # constraint = constraint_ineq
    # for i, con in enumerate(constraint_ineq(variables)):
    #     gradient_active_ineq = [sp.diff(con, var) for var in variables]
    #     print("gradient of active inequal constraint ", i, " is ", sp.latex(gradient_active_ineq))

    # # question 5b
    # variables = [x1, x2]
    # objective_func = function_5b
    # constraint_eq = constraint_5b_eq(variables)
    # constraint_ineq = constraint_5b_ineq(variables)
    # constraint = constraint_eq + constraint_ineq
    # for i, con in enumerate(constraint_ineq):
    #     gradient_active_ineq = [sp.diff(con, var) for var in variables]
    #     print("gradient of active inequal constraint ", i, " is ", sp.latex(gradient_active_ineq))
    # for i, con in enumerate(constraint_eq):
    #     gradient_eq = [sp.diff(con, var) for var in variables]
    #     print("gradient of equal constraint ", i, " is ", sp.latex(gradient_eq))

#    # question 6a
#     variables = [x1, x2]
#     objective_func = function_6a
#     constraint_eq = constraint_6a_ineq(variables)
#     constraint = constraint_eq
#     for i, con in enumerate(constraint_eq):
#         gradient_eq = [sp.diff(con, var) for var in variables]
#         print("gradient of equal constraint ", i, " is ", sp.latex(gradient_eq))

    # # question 6b
    # variables = [x1, x2]
    # objective_func = function_6b
    # constraint_ineq = constraint_6b_ineq(variables)
    # constraint = constraint_ineq
    # for i, con in enumerate(constraint_ineq):
    #     gradient_active_ineq = [sp.diff(con, var) for var in variables]
    #     print("gradient of active inequal constraint ", i, " is ", sp.latex(gradient_active_ineq))
        
    
    # question 6c
    variables = [x1, x2]
    objective_func = function_6c
    constraint_eq = constraint_6c_eq(variables)
    constraint_ineq = constraint_6c_ineq(variables)
    constraint = constraint_eq + constraint_ineq
    for i, con in enumerate(constraint_ineq):
        gradient_active_ineq = [sp.diff(con, var) for var in variables]
        print("gradient of active inequal constraint ", i, " is ", sp.latex(gradient_active_ineq))
    for i, con in enumerate(constraint_eq):
        gradient_eq = [sp.diff(con, var) for var in variables]
        print("gradient of equal constraint ", i, " is ", sp.latex(gradient_eq))
    
    extrema, lagrange_multipliers = lagrange_multiplier_method(objective_func(variables), constraint, variables)
    for i in range(len(extrema)):
        print("Solution ", i)
        print("Extremum found at:", sp.latex(extrema[i]))
        print("Lagrange multipliers:")
        for j, lm in enumerate(lagrange_multipliers[i]):
            print(f"Lambda_{j}: {lm}")   
        print("objective function value:", sp.simplify(objective_func(extrema[i])))
