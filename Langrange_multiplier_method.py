import sympy as sp

def lagrange_multiplier_method(objective_func, constraints, variables):
    # Define Lagrange multiplier symbol
    lambda_ = sp.symbols('lambda')

    # Define Lagrangian function
    lagrangian = objective_func + lambda_ * constraints

    # Compute gradient of the Lagrangian with respect to variables and lambda
    gradient = [sp.diff(lagrangian, var) for var in variables] + [sp.diff(lagrangian, lambda_)]

    # Solve the system of equations
    solution = sp.solve(gradient, variables + [lambda_])

    # Extract extremum and Lagrange multiplier from the solution
    extremum = [solution[var] for var in variables]
    lagrange_multiplier = solution[lambda_]

    return extremum, lagrange_multiplier

# Example usage:
if __name__ == "__main__":
    # Define variables
    x, y = sp.symbols('x y')

    # Define objective function
    objective_func = x**2 + y**2 + 10 * x + 20 * y + 25

    # Define constraint equation
    constraint = x + y

    # Define variables
    variables = [x, y]

    # Find extremum using Lagrange multiplier method
    extremum, lagrange_multiplier = lagrange_multiplier_method(objective_func, constraint, variables)

    print("Extremum found at:", extremum)
    print("Lagrange multiplier:", lagrange_multiplier)
