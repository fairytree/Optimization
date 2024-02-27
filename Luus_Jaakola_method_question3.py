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
            x = x_best + (np.random.rand(dim) - 0.5) * r * (bounds[:, 1] - bounds[:, 0])
            x = np.clip(x, bounds[:, 0], bounds[:, 1])  # Ensure x is within bounds
            x[3] = -(- x[0] * (1.12 + 0.13167 * x[7] - 0.006667 * x[7]**2))
            x[4] = -(- 1.22*x[3] + x[0])
            x[1] = -(- x[0] * x[7] + x[4])
            x[5] = -(- 89 - (x[6]-(86.35 + 1.098 * x[7] - 0.038 * x[7]**2))/0.325)
            x[9] = -133 + 3 * x[6]
            x[8] =  35.82 - 0.222 * x[9] 
            x[2]= 0.001 * x[3] * x[5] * x[8] / (98 - x[5])
            f_new = func(x) + constraints_penalty(x)
            if f_new < f_best:
                x_best, f_best = x, f_new
                path.append(x_best)
                
        # update r
        r = r * search_space_reduction_factor
        print("r: ", r)
        print("x_best: ", x_best)
        i += 1    
    return x_best, f_best, path


