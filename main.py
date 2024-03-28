import sympy as sp
import visualization as vis
import numpy as np
import Newtons_method as newton
import Fletcher_Reeves_method as FRM
import DFP_method as DFP
import Simulated_Annealing_method as SA
import Luus_Jaakola_method as LJ
import Nelder_mead_method as NM
import Gauss_Newton_Marquardt_method as GNM
import matplotlib.pyplot as plt
import math
from colorama import Fore

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

# -----------example-----------
def plot_guess(x_data, y_data, k_opt, message):
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Data')
    T = np.linspace(x_data.min(), x_data.max(), 100)
    Y = k_opt[0]*np.exp(-k_opt[1]*x_data) + k_opt[2]*np.exp(-k_opt[3]*x_data)
    plt.plot(T, Y, 'b-', label='Fit')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title(message)
    print("Optimized parameters:", k_opt)
# Define the residual function
def residual_func_example(x_data, y_data, x):
    return y_data - (x[0]*np.exp(-x[1]*x_data) + x[2]*np.exp(-x[3]*x_data))
# Define the Jacobian function
def jacobian_func_example(x_data, y_data, k):
    J = np.zeros((len(y_data), len(k)))
    J[:, 0] = -np.exp(-k[1]*x_data)
    J[:, 1] = k[0]*x_data*np.exp(-k[1]*x_data)
    J[:, 2] = -np.exp(-k[3]*x_data)
    J[:, 3] = k[2]*x_data*np.exp(-k[3]*x_data)
    return J

# ------Assignment 4 Part A Problem 1 Model B---------
def function_4A1B(k, x):  # k is the vector of parameters, x is the input symbol
    return (k[0]**(-7/10) + (k[1] * k[2] * x / ((1 + k[2] * x)**2))**(-7/10))**(-10/7)
# Calculate residual vector
def residual_vec_4A1B(x_data, y_data, k):
    function_values = function_4A1B(k, x_data)
    residual = function_values - y_data
    return residual
# calculate the norm of the residual vector
def residual_func_4A1B(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4A1B(x_data, y_data, k))
# Define the Jacobian function
def jacobian_func_4A1B(x_data, y_data, k):
    x1 = x_data 
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    J = np.zeros((len(y_data), 3))
    J[:, 0] =  1.0/(k1**1.7*(k1**(-0.7) + (k2*k3*x1/(k3*x1 + 1)**2)**(-0.7))**2.42857142857143)
    J[:, 1] = 1.0/(k2*(k2*k3*x1/(k3*x1 + 1)**2)**0.7*(k1**(-0.7) + (k2*k3*x1/(k3*x1 + 1)**2)**(-0.7))**2.42857142857143)
    J[:, 2] = -1.42857142857143*(k3*x1 + 1)**2*(1.4*k2*k3*x1**2/(k3*x1 + 1)**3 - 0.7*k2*x1/(k3*x1 + 1)**2)/(k2*k3*x1*(k2*k3*x1/(k3*x1 + 1)**2)**0.7*(k1**(-0.7) + (k2*k3*x1/(k3*x1 + 1)**2)**(-0.7))**2.42857142857143)  
    return J

# ------Assignment 4 Part A Problem 1 Model A --------
# define symbolized function:
def function_4A1A(k, x):  # k is the decision variables, x is the input symbol
    R = k[0] + (k[0]**2) * (1 + k[2] * x)**2 / (2 * k[1] * k[2] * x)  
    val = R - (R**2 - k[0]**2)**0.5
    # print("function_value: ", val)
    return val
# Calculate residual vector
def residual_vec_4A1A(x_data, y_data, k):
    function_values = function_4A1A(k, x_data)
    residual = function_values - y_data
    return residual
# Calculate residual vector norm
def residual_func_4A1A(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4A1A(x_data, y_data, k))
# Define the Jacobian function
def jacobian_func_4A1A(x_data, y_data, k):
    x1 = x_data 
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    J = np.zeros((len(y_data), 3))
    J[:, 0] = k1*(k3*x1 + 1)**2/(k2*k3*x1) - (-1.0*k1 + 0.5*(2*k1*(k3*x1 + 1)**2/(k2*k3*x1) + 2)*(k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1))/(-k1**2 + (k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1)**2)**0.5 + 1
    J[:, 1] = 0.5*k1**2*(k3*x1 + 1)**2*(k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1)/(k2**2*k3*x1*(-k1**2 + (k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1)**2)**0.5) - k1**2*(k3*x1 + 1)**2/(2*k2**2*k3*x1)
    J[:, 2] = k1**2*(k3*x1 + 1)/(k2*k3) - k1**2*(k3*x1 + 1)**2/(2*k2*k3**2*x1) - 0.5*(2*k1**2*(k3*x1 + 1)/(k2*k3) - k1**2*(k3*x1 + 1)**2/(k2*k3**2*x1))*(k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1)/(-k1**2 + (k1**2*(k3*x1 + 1)**2/(2*k2*k3*x1) + k1)**2)**0.5  
    return J

# ------Assignment 4 Part A Problem 2---------
def function_4A2(k, x):  # k is the vector of parameters, x is the input symbol
    return k[0] * k[1] * (x[1]**0.5) * x[0] / (k[0] * x[1]**0.5 + x[2] * k[1] * x[0])
# Calculate residual vector
def residual_vec_4A2(x_data, y_data, k):
    function_values = np.zeros(len(x_data))
    for i in range(len(x_data)):
        function_values[i] = function_4A2(k, x_data[i])
    residual = function_values - y_data
    return residual
# calculate the norm of the residual vector
def residual_func_4A2(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4A2(x_data, y_data, k))
# Define the Jacobian function
def jacobian_func_4A2(x_data, y_data, k):
    x1 = x_data[:, 0]
    x2 = x_data[:, 1]
    x3 = x_data[:, 2]
    k1 = k[0]
    k2 = k[1]
    J = np.zeros((len(y_data), 2))
    J[:, 0] = -k1*k2*x1*x2**1.0/(k1*x2**0.5 + k2*x1*x3)**2 + k2*x1*x2**0.5/(k1*x2**0.5 + k2*x1*x3)
    J[:, 1] = -k1*k2*x1**2*x2**0.5*x3/(k1*x2**0.5 + k2*x1*x3)**2 + k1*x1*x2**0.5/(k1*x2**0.5 + k2*x1*x3)
    return J

# ------Assignment 4 Part B Problem 1 Model A --------
# define symbolized function:
def function_4B1A(k, x):  # k is the decision variables, x is the input symbol
    R = k[0] + (k[0]**2) * (1 + k[2] * x)**2 / (2 * k[1] * k[2] * x)  
    val = R - (R**2 - k[0]**2)**(1/2)
    # print("function_value: ", val)
    return val
# Calculate residual vector
def residual_vec_4B1A(x_data, y_data, k):
    function_values = function_4B1A(k, x_data)
    residual = function_values - y_data
    return residual
# Calculate residual vector norm
def residual_func_4B1A(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4B1A(x_data, y_data, k))

# ------Assignment 4 Part A Problem 1 Model B---------
def function_4B1B(k, x):  # k is the vector of parameters, x is the input symbol
    return (k[0]**(-0.7) + (k[1] * k[2] * x / ((1 + k[2] * x)**2))**(-0.7))**(-10/7)
# Calculate residual vector
def residual_vec_4B1B(x_data, y_data, k):
    function_values = function_4B1B(k, x_data)
    residual = function_values - y_data
    return residual
# calculate the norm of the residual vector
def residual_func_4B1B(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4B1B(x_data, y_data, k))

# ------Assignment 4 Part B Problem 2---------
def function_4B2(k, x):  # k is the vector of parameters, x is the input symbol
    return k[0] * k[1] * (x[1]**0.5) * x[0] / (k[0] * x[1]**0.5 + x[2] * k[1] * x[0])
# Calculate residual vector
def residual_vec_4B2(x_data, y_data, k):
    function_values = np.zeros(len(x_data))
    for i in range(len(x_data)):
        function_values[i] = function_4B2(k, x_data[i])
    residual = function_values - y_data
    return residual
# calculate the norm of the residual vector
def residual_func_4B2(x_data, y_data, k):
    return np.linalg.norm(residual_vec_4B2(x_data, y_data, k))

if __name__ == "__main__":
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, k1, k2, k3, y1, f1 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 k1 k2 k3 y1 f1')
    
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
    # cooling_rate = 0.80  # Cooling rate
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
    # search_space_reduction_factor = 0.90
    # function = function_2
    # constraint = constraint_2
    # extremum, best_cost, path = LJ.luus_jaakola_optimize(function, initial_guess, r, bounds, constraint, innerloop_iterations,outloop_iterations, search_space_reduction_factor)
    # print("extremum:", extremum)
    # print("Objective function value at extremumn:", best_cost)


    # # ------Assignment 3 question 3 --------
    # variables = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    # bounds = np.array([[0.01, 2000], [0.01, 16000], [0.01, 120], [0.01, 5000], [0.01, 2000], [85, 93], [90, 95], [3, 12], [1.2, 4], [145, 162]])
    # innerloop_iterations = 10000
    # outloop_iterations = 100
    # r = 1
    # initial_guess = [500, 5000, 30, 1000, 700, 30, 6.0, 8.0, 0.1, 50.0]
    # search_space_reduction_factor = 0.95
    # function = function_3
    # constraint = constraint_3
    # extremum, best_cost, path = LJ.luus_jaakola_optimize_question3(function, initial_guess, r, bounds, constraint, innerloop_iterations,outloop_iterations, search_space_reduction_factor)
    # print("extremum:", extremum)
    # print("Objective function value at extremumn:", best_cost)
    
        
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
    
    # # ----------example--------------------
    # x_data = np.linspace(-10, 10, 100)
    # y_true = 2.0 * np.exp(-0.5 * x_data) + 1.5 * np.exp(-0.2 * x_data)
    # np.random.seed(0)
    # y_data = y_true + np.random.normal(0, 0.01, size=x_data.shape)
    # initial_guess = np.array([-1.5, 0.8, 1.2, 0.3])
    # residual_func = residual_func_example
    # jacobian_func = jacobian_func_example
    # k_opt, path = GNM.gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func)
    # plt.figure()
    # vis.plot_guess(x_data, y_data, initial_guess, "initial guess")
    # plt.figure()
    # vis.plot_guess(x_data, y_data, k_opt, "last Iteration")
    # plt.show()
    
    # # ------Assignment 4 Part A Problem 1 Model A--------
    # variables = sp.Matrix([k1, k2, k3])
    # x = x1
    # F = sp.Matrix([function_4A1A(variables, x)])
    # J_matrix = F.jacobian(variables)
    # print(Fore.RED + "Jacobian Matrix" + Fore.RESET)
    # print(f"1: {J_matrix[0]}")
    # print(f"2: {J_matrix[1]}")
    # print(f"3: {J_matrix[2]}")
    # print(Fore.RED + "^" + Fore.RESET)
    # initial_guess = np.array([81.7 * 0.01, 7.89 * 0.01, 53.5 * 0.01])
    # # data for temperatur at 600
    # x_data = np.array([1.0, 7.0, 4.0, 10.0, 14.6, 5.5, 8.5, 3.0, 0.22, 1.0])
    # y_data = np.array([0.0392, 0.0416, 0.0416, 0.0326, 0.0247, 0.0415, 0.0376, 0.0420, 0.0295, 0.0410])
    # # data for temperature at 575
    # # x_data = np.array([1.0, 3.0, 5.0, 7.0, 9.6])
    # # y_data = np.array([0.0227, 0.0277, 0.0255, 0.0217, 0.0183])
    # residual_func = residual_vec_4A1A
    # jacobian_func = jacobian_func_4A1A
    # k_opt, path = GNM.gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func)
    # # visualization
    # vis.data_table(lambda k: residual_func_4A1A(x_data, y_data, k), None, ["k_H", "k_R", "K_A"], path)
    # x_values = np.linspace(0.5, 15.0, 200)
    # function_values = function_4A1A(k_opt, x_values)
    # figure, axes = plt.subplots()
    # axes.plot(x_values, function_values, "r-")
    # axes.plot(x_data, y_data, "bo")
    # axes.set_xlabel("Pressure")
    # axes.set_ylabel("Initial Rate")
    # plt.show()

    # # ------Assignment 4 Part A Problem 1 Model B--------
    # variables = sp.Matrix([k1, k2, k3])
    # x = x1
    # F = sp.Matrix([function_4A1B(variables, x)])
    # J_matrix = F.jacobian(variables)
    # print(Fore.RED + "Jacobian Matrix" + Fore.RESET)
    # print(f"1: {J_matrix[0]}")
    # print(f"2: {J_matrix[1]}")
    # print(f"3: {J_matrix[2]}")
    # print(Fore.RED + "^" + Fore.RESET)
    # initial_guess = np.array([81.7 * 0.01, 7.89 * 0.01, 53.5 * 0.01])
    # # data for temperatur at 600F
    # x_data = np.array([1.0, 7.0, 4.0, 10.0, 14.6, 5.5, 8.5, 3.0, 0.22, 1.0])
    # y_data = np.array([0.0392, 0.0416, 0.0416, 0.0326, 0.0247, 0.0415, 0.0376, 0.0420, 0.0295, 0.0410])
    # # # data for temperature 575F
    # # x_data = np.array([1.0, 3.0, 5.0, 7.0, 9.6])
    # # y_data = np.array([0.0227, 0.0277, 0.0255, 0.0217, 0.0183])
    # residual_func = residual_vec_4A1B
    # jacobian_func = jacobian_func_4A1B
    # k_opt, path = GNM.gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func)
    # # visualization
    # vis.data_table(lambda k: residual_func_4A1B(x_data, y_data, k), None, ["k_H", "k_R", "K_A"], path)
    # x_values = np.linspace(0.5, 15.0, 200)
    # function_values = function_4A1B(k_opt, x_values)
    # figure, axes = plt.subplots()
    # axes.plot(x_values, function_values, "r-")
    # axes.plot(x_data, y_data, "bo")
    # axes.set_xlabel("Pressure")
    # axes.set_ylabel("Initial Rate")
    # plt.show()

    # # ------Assignment 4 Part A Problem 2--------
    # params = sp.Matrix([k1, k2])
    # x = sp.Matrix([x1, x2, x3])
    # F = sp.Matrix([function_4A2(params, x)])
    # J_matrix = F.jacobian(params)
    # print(Fore.RED + "Jacobian Matrix" + Fore.RESET)
    # print(f"1: {J_matrix[0]}")
    # print(f"2: {J_matrix[1]}")
    # print(Fore.RED + "^" + Fore.RESET)
    # initial_guess = np.array([1.334, 0.611])
    # # data
    # x_data = np.array([[3.05, 3.07, 0.658], [1.37, 3.18, 0.439], [3.17, 1.24, 0.452], [3.02, 3.85, 0.695], [4.31, 3.15, 0.635], [2.78, 3.89, 0.670], [3.11, 6.48, 0.760], [2.96, 3.13, 0.642], [2.84, 3.14, 0.665], [1.46, 7.93, 0.525], [1.38, 7.79, 0.483], [1.42, 8.03, 0.522], [1.49, 7.78, 0.530], [3.01, 3.03, 0.635], [1.35, 8.00, 0.480], [1.52, 8.22, 0.544], [5.95, 6.13, 0.893], [1.46, 8.41, 0.517], [5.68, 7.75, 0.996], [1.36, 3.10, 0.416], [1.42, 1.25, 0.367], [3.18, 7.89, 0.835], [2.87, 3.06, 0.609]])
    # y_data = np.array([2.73, 2.86, 3.00, 2.64, 2.60, 2.73, 2.56, 2.69, 2.77, 2.91, 2.87, 2.97, 2.93, 2.75, 2.90, 2.94, 2.38, 2.89, 2.41, 2.81, 2.86, 2.59, 2.76])
    # residual_func = residual_vec_4A2
    # jacobian_func = jacobian_func_4A2
    # k_opt, path = GNM.gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func, step_size_init=0.1, lam=0.1, tune="lambda")
    # # visualization
    # vis.data_table(lambda k: residual_func_4A2(x_data, y_data, k), None, ["k_O", "k_P"], path)

    # #------Assignment 4 Part B Problem 1 Model A--------
    # initial_guess = np.array([7.89 * 0.01, 81.7 * 0.01, 53.5 * 0.01])
    # r = 0.2
    # bounds = np.array([[-50, 50], [-150, 150], [-100, 100]])
    # # data for temperatur at 600
    # x_data = np.array([1.0, 7.0, 4.0, 10.0, 14.6, 5.5, 8.5, 3.0, 0.22, 1.0])
    # y_data = np.array([0.0392, 0.0416, 0.0416, 0.0326, 0.0247, 0.0415, 0.0376, 0.0420, 0.0295, 0.0410])
    # # data for temperature at 575
    # # x_data = np.array([1.0, 3.0, 5.0, 7.0, 9.6])
    # # y_data = np.array([0.0227, 0.0277, 0.0255, 0.0217, 0.0183])
    # residual_func = residual_vec_4B1A
    # k_opt, _, path = LJ.luus_jaakola_optimize(lambda k: residual_func_4B1A(x_data, y_data, k), initial_guess, r, bounds)
    # print(Fore.GREEN + "Final Residual Vector" + Fore.RESET)
    # print(residual_vec_4B1A(x_data, y_data, k_opt))
    # print(Fore.GREEN + "^" + Fore.RESET)
    # print(Fore.GREEN + "Final Residual" + Fore.RESET)
    # print(residual_func_4B1A(x_data, y_data, k_opt))
    # print(Fore.GREEN + "^" + Fore.RESET)
    # # visualization
    # vis.data_table(lambda k: residual_func_4B1A(x_data, y_data, k), None, ["k_H", "k_R", "K_A"], path)
    # x_values = np.linspace(0.5, 15.0, 200)
    # function_values = function_4B1A(k_opt, x_values)
    # figure, axes = plt.subplots()
    # axes.plot(x_values, function_values, "r-")
    # axes.plot(x_data, y_data, "bo")
    # axes.set_xlabel("Pressure")
    # axes.set_ylabel("Initial Rate")
    # plt.show()

    # # # ------Assignment 4 Part B Problem 1 Model B--------
    # initial_guess = np.array([9.5 * 0.01, 62.8 * 0.01, 51.5 * 0.01])
    # r = 0.2
    # bounds = np.array([[-50, 50], [-150, 150], [-100, 100]])
    # # data for temperatur at 600
    # x_data = np.array([1.0, 7.0, 4.0, 10.0, 14.6, 5.5, 8.5, 3.0, 0.22, 1.0])
    # y_data = np.array([0.0392, 0.0416, 0.0416, 0.0326, 0.0247, 0.0415, 0.0376, 0.0420, 0.0295, 0.0410])
    # # data for temperature at 575
    # # x_data = np.array([1.0, 3.0, 5.0, 7.0, 9.6])
    # # y_data = np.array([0.0227, 0.0277, 0.0255, 0.0217, 0.0183])
    # residual_func = residual_vec_4B1B
    # k_opt, _, path = LJ.luus_jaakola_optimize(lambda k: residual_func_4B1B(x_data, y_data, k), initial_guess, r, bounds)
    # print(Fore.GREEN + "Final Residual Vector" + Fore.RESET)
    # print(residual_vec_4B1B(x_data, y_data, k_opt))
    # print(Fore.GREEN + "^" + Fore.RESET)
    # print(Fore.GREEN + "Final Residual" + Fore.RESET)
    # print(residual_func_4B1B(x_data, y_data, k_opt))
    # print(Fore.GREEN + "^" + Fore.RESET)
    # # visualization
    # vis.data_table(lambda k: residual_func_4B1B(x_data, y_data, k), None, ["k_H", "k_R", "K_A"], path)
    # x_values = np.linspace(0.5, 15.0, 200)
    # function_values = function_4B1B(k_opt, x_values)
    # figure, axes = plt.subplots()
    # axes.plot(x_values, function_values, "r-")
    # axes.plot(x_data, y_data, "bo")
    # axes.set_xlabel("Pressure")
    # axes.set_ylabel("Initial Rate")
    # plt.show()

    # # ------Assignment 4 Part B Problem 2--------
    initial_guess = np.array([1.334, 0.611])
    r = 0.2
    bounds = np.array([[-100, 100], [-100, 100]])
    # data
    x_data = np.array([[3.05, 3.07, 0.658], [1.37, 3.18, 0.439], [3.17, 1.24, 0.452], [3.02, 3.85, 0.695], [4.31, 3.15, 0.635], [2.78, 3.89, 0.670], [3.11, 6.48, 0.760], [2.96, 3.13, 0.642], [2.84, 3.14, 0.665], [1.46, 7.93, 0.525], [1.38, 7.79, 0.483], [1.42, 8.03, 0.522], [1.49, 7.78, 0.530], [3.01, 3.03, 0.635], [1.35, 8.00, 0.480], [1.52, 8.22, 0.544], [5.95, 6.13, 0.893], [1.46, 8.41, 0.517], [5.68, 7.75, 0.996], [1.36, 3.10, 0.416], [1.42, 1.25, 0.367], [3.18, 7.89, 0.835], [2.87, 3.06, 0.609]])
    y_data = np.array([2.73, 2.86, 3.00, 2.64, 2.60, 2.73, 2.56, 2.69, 2.77, 2.91, 2.87, 2.97, 2.93, 2.75, 2.90, 2.94, 2.38, 2.89, 2.41, 2.81, 2.86, 2.59, 2.76])
    residual_func = residual_vec_4B2
    k_opt, _, path = LJ.luus_jaakola_optimize(lambda k: residual_func_4B2(x_data, y_data, k), initial_guess, r, bounds)
    print(Fore.GREEN + "Final Residual Vector" + Fore.RESET)
    print(residual_vec_4B2(x_data, y_data, k_opt))
    print(Fore.GREEN + "^" + Fore.RESET)
    print(Fore.GREEN + "Final Residual" + Fore.RESET)
    print(residual_func_4B2(x_data, y_data, k_opt))
    print(Fore.GREEN + "^" + Fore.RESET)
    # visualization
    vis.data_table(lambda k: residual_func_4B2(x_data, y_data, k), None, ["k_O", "k_P"], path)
