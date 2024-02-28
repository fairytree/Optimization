import numpy as np

def nelder_mead(func, initial_simplex, tol=1e-5, max_iter=500):
    num_vertices = len(initial_simplex)
    rho = 1
    chi = 2
    gamma = 0.5
    sigma = 0.5
    path = [initial_simplex[0]]
    
    simplex = initial_simplex

    for i in range(max_iter):
        simplex = sorted(simplex, key=lambda x: func(x))
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + rho * (centroid - simplex[-1])
        if np.all(func(simplex[0]) <= func(xr)) and np.all(func(xr) < func(simplex[-2])):
            simplex[-1] = xr
        elif np.all(func(xr) < func(simplex[0])):
            # Expansion
            xe = centroid + chi * (xr - centroid)
            simplex[-1] = xe if np.all(func(xe) < func(xr)) else xr
        else:
            # Contraction
            xc = centroid + gamma * (simplex[-1] - centroid)
            if np.all(func(xc) < func(simplex[-1])):
                simplex[-1] = xc
            else:
                # Shrink
                simplex = [simplex[0]] + [simplex[0] + sigma * (x - simplex[0]) for x in simplex[1:]]
        
        path.append(simplex[0])
        # print("simplex[0] is: ", simplex[0])
        # print("simplex: ", simplex)
        # print("func(simplex[0]) is: ", func(simplex[0]))
        # print("func(simplex) is: ", func(simplex))
        if np.all(np.abs(func(simplex[0]) - func(simplex)) < tol):
            break

    return simplex[0], func(simplex[0]), path