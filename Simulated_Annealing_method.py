import math
import random

def simulated_annealing(obj_func, eq_constraint, ineq_constraint, initial_solution, initial_temperature, cooling_rate, max_iterations):
    current_solution = initial_solution
    current_cost = obj_func(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature
    path = [initial_solution] 
    
    for i in range(max_iterations):
        neighbor_solution = generate_neighbor(current_solution, step_size=0.1)
        if all(constraint(neighbor_solution) <= 0 for constraint in ineq_constraint):
            neighbor_cost = obj_func(neighbor_solution)
            if acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
                    path.append(best_solution)
                    
        temperature *= cooling_rate

    return best_solution, best_cost, path

def generate_neighbor(x, step_size):
    neighbor = [xi + random.uniform(-step_size, step_size) for xi in x]
    return neighbor

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)