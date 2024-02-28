import sympy as sp
import visualization as vis
import numpy as np
import Newtons_method as newton
import Fletcher_Reeves_method as FRM
import DFP_method as DFP
import Simulated_Annealing_method as SA
import Luus_Jaakola_method as LJ
import Nelder_mead_method as NM
import math

# ------Assignment 2--------
# question 1a-1 Newton's method
def function_1a1(x):
    return 1 + x[0] + x[1] + x[2] + x[3] + x[0] * x[1] + x[0] * x[2] + x[0] * x[3] + x[1] * x[2] + x[1] * x[3] + x[2] * x[3] + x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

# ------Assignment 2--------
# question 1a-2 Newton's method
def function_1a2(x):
    return 8 * x[0]**2 + 4 * x[0] * x[1] + 5 * x[1]**2

# ------Assignment 2--------
# question 1b Fletcher_Reeves Method 
def function_1b(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2

# ------Assignment 2--------
# question 1c DFP method
def function_1c(x):
    return x[0] - x[1] + 2 * x[0]**2 + 2 * x[0] * x[1] + x[1]**2

# ------Assignment 3 question 1 --------
def function_1(x):
    return x[0]**2 + x[1]**2 - 0.3 * math.cos(3 * math.pi * x[0]) - 0.4 * math.cos(4 * math.pi * x[1]) + 0.7
def constraint_1_ineq(x):
    return [
        lambda x: -1 - x[0],
        lambda x: x[0] - 1,
        lambda x: -1 - x[1],
        lambda x: x[1] - 1
    ]
    

# ------Assignment 3 question 2 --------
def function_2(x):
    n = 10
    P = 750
    f = 0
    w = [-10.021, -21.096, -37.986, -9.846, -28.653, -18.918, -28.032, -14.640, -30.594, -26.111]
    sum_x = sum(x)
    epsilon = 1e-18  # A small constant to avoid log(0)
    for i in range(n):
        f += x[i] * (w[i] + np.log(P) + np.log((x[i] + epsilon) / (sum_x + n*epsilon)))
    return f
def constraint_2(x):
    penalties = 0
    epsilon = 0.01  # Tolerance for equality
    # Apply penalties for constraint violations
    multiplier = 500
    if abs(x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2) > epsilon:
        penalties += abs(x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2) * multiplier
    if abs(x[3] + 2 * x[4] + x[5] + x[6] - 1) > epsilon:
        penalties += abs(x[3] + 2 * x[4] + x[5] + x[6] - 1) * multiplier
    if abs(x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1) > epsilon:
        penalties += abs(x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1) * multiplier
    return penalties


# ------Assignment 3 question 3 --------
def function_3(x):
    return -(0.063 * x[3] * x[6] - 5.04 * x[0] - 0.035 * x[1] - 10 * x[2] - 3.36 * x[4])
def constraint_3(x):
    penalty_value = 0
    multiplier = 500
    penalty_value += multiplier * max(0, x[0]-2000)
    penalty_value += multiplier * max(0, 0.01 - x[0])
    
    penalty_value += multiplier * max(0, x[1]-16000)
    penalty_value += multiplier * max(0, 0.01 - x[1])
    
    penalty_value += multiplier * max(0, x[2]-120)
    penalty_value += multiplier * max(0, 0.01 - x[2])
    
    penalty_value += multiplier * max(0, x[3]-5000)
    penalty_value += multiplier * max(0, 0.01 - x[3])
    
    penalty_value += multiplier * max(0, x[4]-2000)
    penalty_value += multiplier * max(0, 0.01 - x[4])
    
    penalty_value += multiplier * max(0, x[5]-93)
    penalty_value += multiplier * max(0, 85 - x[5])
    
    penalty_value += multiplier * max(0, x[6]-95)
    penalty_value += multiplier * max(0, 90 - x[6])
    
    penalty_value += multiplier * max(0, x[7]-12)
    penalty_value += multiplier * max(0, 3 - x[7])
    
    penalty_value += multiplier * max(0, x[8]-4)
    penalty_value += multiplier * max(0, 1.2 - x[8])
    
    penalty_value += multiplier * max(0, x[9]-162)
    penalty_value += multiplier * max(0, 145 - x[9])
    
    epsilon = 0.1  # Tolerance for equality
    multiplier_equal = 500
    if abs(x[3] - x[0] * (1.12 + 0.13167 * x[7] - 0.006667 * x[7]**2)) > epsilon:
        penalty_value += abs(x[3] - x[0] * (1.12 + 0.13167 * x[7] - 0.006667 * x[7]**2)) * multiplier_equal
        print("hello")
    if abs(x[4] - 1.22*x[3] + x[0]) > epsilon:
        penalty_value += abs(x[4] - 1.22*x[3] + x[0]) * multiplier_equal
    if abs(x[1] - x[0] * x[7] + x[4]) > epsilon:
        penalty_value += abs(x[1] - x[0] * x[7] + x[4]) * multiplier_equal
    if abs(x[5] - 89 - (x[6]-(86.35 + 1.098 * x[7] - 0.038 * x[7]**2))/0.325) > epsilon:
        penalty_value += abs(x[5] - 89 - (x[6]-(86.35 + 1.098 * x[7] - 0.038 * x[7]**2))/0.325) * multiplier_equal
    if abs(x[9] + 133 - 3 * x[6]) > epsilon:
        penalty_value += abs(x[9] + 133 - 3 * x[6]) * multiplier_equal
    if abs(x[8] - 35.82 + 0.222 * x[9]) > epsilon:
        penalty_value += abs(x[8] - 35.82 + 0.222 * x[9]) * multiplier_equal
    if abs(x[2] - 0.001 * x[3] * x[5] * x[8] / (98 - x[5])) > epsilon:
        penalty_value += abs(x[2] - 0.001 * x[3] * x[5] * x[8] / (98 - x[5])) * multiplier_equal
        
    return penalty_value


# ------Assignment 3 question 4 --------
def function_4(x):
    return 100 * (x[1] - x[0]**2)**2 + (1- x[0])**2
def constraint_4(x):
    return 0
def modified_objective_4(x):
    return function_4(x) + constraint_4(x)

# ------Assignment 3 question 5 --------
def function_5(x):
    return (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[3])**4
def constraint_5(x):
    return 0
def modified_objective_5(x):
    return function_5(x) + constraint_5(x)

if __name__ == "__main__":
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    
    # #-----question 1a-1 Newton's Method with initial_guess = sp.Matrix([-3.0, -30, -4, -0.1])-----
    # variables = [x1, x2, x3, x4]
    # initial_guess = sp.Matrix([-3.0, -30.0, -4.0, -0.1])
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
    # variables = [x1, x2]
    # initial_guess = np.array([0.0, 0.0])  # Initial point, this is a 1D array
    # function = function_1c
    # extremum, path = DFP.dfp_method(function(variables), variables, initial_guess)
    
    # # ------Assignment 3 question 1 --------
    # variables = [x1, x2]
    # initial_guess = [0.2, 0.2]  # Initial guess
    # function = function_1
    # constraint_inequal = constraint_1_ineq(initial_guess)  # Adjusted to apply constraints correctly
    # initial_temperature = 10000  # Initial temperature
    # cooling_rate = 0.90  # Cooling rate
    # max_iterations = 1000000  # Maximum number of iterations
    # extremum, best_cost, path = SA.simulated_annealing(function, lambda x: True, constraint_inequal, initial_guess, initial_temperature, cooling_rate, max_iterations)
    # print("best_cost: ", best_cost)
    
    
    # # ------Assignment 3 question 2 --------
    # variables = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    # bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    # innerloop_iterations = 10000
    # outloop_iterations = 100
    # r = 0.2
    # initial_guess = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # search_space_reduction_factor = 0.95
    # function = function_2
    # constraint = constraint_2
    # extremum, best_cost, path = LJ.luus_jaakola_optimize(function, initial_guess, r, bounds, constraint, innerloop_iterations,outloop_iterations, search_space_reduction_factor)
    # print("extremum:", extremum)
    # print("Objective function value at extremumn:", best_cost)


    # ------Assignment 3 question 3 --------
    variables = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    bounds = np.array([[0.01, 2000], [0.01, 16000], [0.01, 120], [0.01, 5000], [0.01, 2000], [85, 93], [90, 95], [3, 12], [1.2, 4], [145, 162]])
    innerloop_iterations = 10000
    outloop_iterations = 100
    r = 1
    initial_guess = [500, 5000, 30, 1000, 700, 30, 6.0, 8.0, 0.1, 50.0]
    search_space_reduction_factor = 0.95
    function = function_3
    constraint = constraint_3
    extremum, best_cost, path = LJ.luus_jaakola_optimize_question3(function, initial_guess, r, bounds, constraint, innerloop_iterations,outloop_iterations, search_space_reduction_factor)
    print("extremum:", extremum)
    print("Objective function value at extremumn:", best_cost)
    
        
    # # ------Assignment 3 question 4 --------
    # variables = [x1, x2]
    # start_point = np.array([-1.2, 1.0])
    # step_size = 0.1
    # point1 = start_point
    # point2 = start_point + np.array([step_size, 0])
    # point3 = start_point + np.array([0, step_size])
    # initial_simplex = [point1, point2, point3]
    # function = modified_objective_4
    # extremum, opt_value, path = NM.nelder_mead(function, initial_simplex)

#    # ------Assignment 3 question 5 --------
#     variables = [x1, x2, x3, x4]
#     start_point = np.array([3.0, -1.0, 0.0, 1.0])
#     step_size = 0.1
#     point1 = start_point
#     point2 = start_point + np.array([step_size, 0, 0, 0])
#     point3 = start_point + np.array([0, step_size, 0, 0])
#     point4 = start_point + np.array([0, 0, step_size, 0])
#     point5 = start_point + np.array([0, 0, 0, step_size])
#     initial_simplex = [point1, point2, point3, point4, point5]
#     function = modified_objective_5
#     extremum, opt_value, path = NM.nelder_mead(function, initial_simplex)
     

    vis.data_table(function, extremum, variables, path)
    vis.plot(function, path, variables)
        