import numpy as np
import math
import random
import sympy as sp

def simulated_annealing(obj_func, eq_constraint, ineq_constraint, initial_solution, initial_temperature, cooling_rate, max_iterations):
    current_solution = initial_solution
    current_cost = obj_func(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature

    for i in range(max_iterations):
        print("i: ", i)
        neighbor_solution = generate_neighbor(current_solution, step_size=0.1)
        print("neighbor_solution: ", neighbor_solution)
        print("eq_constraint(neighbor_solution): ", eq_constraint(neighbor_solution))
        print("ineq_constraint(neighbor_solution): ", ineq_constraint(neighbor_solution))
        if eq_constraint(neighbor_solution) == 0 and ineq_constraint(neighbor_solution) <= 0:
            neighbor_cost = obj_func(neighbor_solution)
            print("j")
            if acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                print("k")
                if current_cost < best_cost:
                    print("h")
                    best_solution = current_solution
                    best_cost = current_cost

        temperature *= cooling_rate

    return best_solution, best_cost


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
        f += x[i] * (w[i] + sp.log(P) + sp.log(x[i]/sum_x))
    return f
def constraint_3_eq(x):
    return [x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2, x[3] + 2 * x[4] + x[5] + x[6] - 1, x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1]
def constraint_3_ineq(x):
    return [0 * x[0]]

def generate_neighbor(x, step_size):
    # Generate a random neighbor within a certain step size
    neighbor = [xi + random.uniform(-step_size, step_size) for xi in x]
    return neighbor

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)
    

# Example usage:
if __name__ == "__main__":
    # Define variables
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    
    # Example usage
    initial_solution = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  # Initial guess
    initial_temperature = 1000  # Initial temperature
    cooling_rate = 0.95  # Cooling rate
    max_iterations = 1000  # Maximum number of iterations

    # question 3
    objective_function = function_3
    constraint_equal = constraint_3_eq
    constraint_inequal = constraint_3_ineq

    best_solution, best_cost = simulated_annealing(objective_function, constraint_equal, constraint_inequal, initial_solution, initial_temperature, cooling_rate, max_iterations)
    print("Best solution found:", best_solution)
    print("Objective function value at best solution:", best_cost)
