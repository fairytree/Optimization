import numpy as np

def simulated_annealing(objective_func, initial_solution, temperature=100, cooling_rate=0.95, max_iterations=1000):
    current_solution = initial_solution
    current_energy = objective_func(*initial_solution)
    best_solution = current_solution
    best_energy = current_energy

    for iteration in range(max_iterations):
        # Adjust temperature
        temperature *= cooling_rate
        
        # Generate a new solution by perturbing the current solution
        new_solution = current_solution + np.random.normal(size=len(initial_solution))
        
        # Evaluate the energy (objective function value) of the new solution
        new_energy = objective_func(*new_solution)
        
        # If the new solution is better, accept it
        if new_energy < current_energy:
            current_solution = new_solution
            current_energy = new_energy
            # Update the best solution if needed
            if new_energy < best_energy:
                best_solution = new_solution
                best_energy = new_energy
        else:
            # If the new solution is worse, accept it probabilistically
            probability = np.exp((current_energy - new_energy) / temperature)
            if np.random.rand() < probability:
                current_solution = new_solution
                current_energy = new_energy

    return best_solution, best_energy

# Example usage:
if __name__ == "__main__":
    # Define your multivariable objective function
    def objective_func(x, y):
        return x**2 + y**2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7
    
    # Initial solution
    initial_solution = np.array([0.0, 0.0])

    # Find extremum using simulated annealing
    extremum, energy = simulated_annealing(objective_func, initial_solution)

    print("Extremum found at:", extremum)
    print("Objective function value at extremum:", energy)
