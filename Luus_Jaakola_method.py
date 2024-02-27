import numpy as np

# Luus-Jaakola optimization method
def luus_jaakola_optimize(func, initial_guess, r, bounds, constraints_penalty, innerloop_iterations, outloop_iterations, search_space_reduction_factor):
    dim = len(bounds)
    x_best = initial_guess    
    f_best = func(x_best) + constraints_penalty(x_best)
    path = [initial_guess]
    
    for i in range (outloop_iterations):    
        for iteration in range(innerloop_iterations):
            # Generate new solution within bounds
            x_new = x_best + (np.random.rand(dim) - 0.5) * r
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])  # Ensure x_new is within bounds
            f_new = func(x_new) + constraints_penalty(x_new)
            if f_new < f_best:
                x_best, f_best = x_new, f_new
                path.append(x_best)
                
        # update r
        r = r * search_space_reduction_factor
        print("r: ", r)
        print("x_best: ", x_best)
        i += 1    
    return x_best, f_best, path


