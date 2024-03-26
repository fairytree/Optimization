import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func, lam = 0.001, max_iter_outer=1000, max_iter_inner=10, tol=1e-16):
       
    k = initial_guess   
    k_old = initial_guess
    residual = residual_func(x_data, y_data, k)
    print(residual)
    for j in range(max_iter_outer):
        residual = residual_func(x_data, y_data, k)
        J = jacobian_func(x_data, y_data, k)

        # Calculate the Gauss-Newton step with Marquardt modification: (J.T * J + lambda * I) * delta = - J.T * residual
        matrix_A = J.T.dot(J) + lam * np.eye((J.T).shape[0])
        vector_b = -J.T.dot(residual)
        delta = np.linalg.lstsq(matrix_A, vector_b, rcond=None)[0]
        print("delta: ", delta)

        
        step = 0.1
        # Update the parameters
        k_new = k + step * delta
        # Calculate the norm of residual (cost function)
        ssr_old = np.linalg.norm(residual)
        ssr_new = np.linalg.norm(residual_func(x_data, y_data, k_new))
        print("ssr_old: ", ssr_old, "ssr_new: ", ssr_new)
        
        if ssr_new < ssr_old:
            k_old = k
            lam *= 0.1
            k = k_new 
            print("found step: ", step, "delta: ", delta)    

        else:
            lam *= 10
        
        # Use bisection rule to obtain step size lambda
        # step = 1
        # for i in range(max_iter_inner):
        #     # Update the parameters
        #     k_new = k + step * delta
        #     # Calculate the norm of residual (cost function)
        #     ssr_old = (np.linalg.norm(residual))**2
        #     ssr_new = (np.linalg.norm(residual_func(x_data, y_data, k_new)))**2
        #     # print("ssr_old: ", ssr_old, "ssr_new: ", ssr_new)
        #     if ssr_new < ssr_old:
        #         k_old = k
        #         k = k_new 
        #         print("found step: ", step, "delta: ", delta)           
        #         break
        #     else:
        #         step /= 2
    
        # Check convergence
        if np.mean(np.abs(delta/k_old)) < tol:
            print(f"Converged in {j+1} iterations.")
            print('k:', k)
            break
    else:
        print("Did not converge.")
        print('k:', k)
    
    return k