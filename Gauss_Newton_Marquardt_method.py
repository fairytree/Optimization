import numpy as np
from colorama import Fore

def gauss_newton_marquardt(initial_guess, x_data, y_data, residual_func, jacobian_func, step_size_init=0.1, lam=0.01, max_iter_outer=5000, max_iter_inner=100, tol=1e-16, tune="lambda"):
       
    k = initial_guess   
    k_old = initial_guess
    residual = residual_func(x_data, y_data, k)
    print(Fore.GREEN + "Initial Residual Vector" + Fore.RESET)
    print(residual)
    print(Fore.GREEN + "^" + Fore.RESET)
    print(Fore.GREEN + "Initial Residual" + Fore.RESET)
    print(np.linalg.norm(residual))
    print(Fore.GREEN + "^" + Fore.RESET)
    path = [k]
    for j in range(max_iter_outer):
        residual = residual_func(x_data, y_data, k)
        J = jacobian_func(x_data, y_data, k)

        # Calculate the Gauss-Newton step with Marquardt modification: (J.T * J + lambda * I) * delta = - J.T * residual
        matrix_A = J.T @ J + lam * np.eye(J.shape[1])
        vector_b = -J.T @ residual
        delta = np.linalg.inv(matrix_A) @ vector_b
        # delta = np.linalg.lstsq(matrix_A, vector_b, rcond=None)[0]
        # print("delta: ", delta)

        step_size = step_size_init
        if tune == "lambda":
            # Update the parameters and append to path
            k_new = k + step_size * delta
            path.append(k_new)
            # Calculate the norm of residual (cost function)
            ssr_old = np.linalg.norm(residual)
            ssr_new = np.linalg.norm(residual_func(x_data, y_data, k_new))
            # print("ssr_old: ", ssr_old, "ssr_new: ", ssr_new)
            
            if ssr_new < ssr_old:
                k_old = k
                lam *= 0.5
                k = k_new 
                # print("found step_size: ", step_size, "delta: ", delta)    

            else:
                lam *= 5
        
        # deprecated
        elif tune == "step_size":
            # TODO test bisection rule to obtain step_size lambda
            step_size = 1
            for i in range(max_iter_inner):
                # Update the parameters and append to path
                k_new = k + step_size * delta
                # Calculate the norm of residual (cost function)
                ssr_old = np.linalg.norm(residual)
                ssr_new = np.linalg.norm(residual_func(x_data, y_data, k_new))
                # print("ssr_old: ", ssr_old, "ssr_new: ", ssr_new)
                if ssr_new < ssr_old:
                    k_old = k
                    k = k_new
                    path.append(k_new)
                    # print("found step_size: ", step_size, "delta: ", delta)           
                    break
                else:
                    step_size /= 2

        else:
            raise ValueError(f"Expected 'lambda' or 'step_size' for parameter 'tune' in method 'Gauss_Newton_Marquardt_method.gauss_newton_marquartd()', but got {tune}")
    
        # Check convergence
        if np.mean(np.abs(delta/k_old)) < tol:
            print(Fore.GREEN + "Final Residual Vector" + Fore.RESET)
            print(residual_func(x_data, y_data, k))
            print(Fore.GREEN + "^" + Fore.RESET)
            print(Fore.GREEN + "Final Residual" + Fore.RESET)
            print(np.linalg.norm(residual_func(x_data, y_data, k)))
            print(Fore.GREEN + "^" + Fore.RESET)
            print(Fore.BLUE + "Parameters" + Fore.RESET)
            print(k)
            print(Fore.BLUE + "^" + Fore.RESET)
            print(f"Converged in {j+1} iterations.")
            print(f"Last λ: {lam}")
            print(f"Last step_size: {step_size}")
            break
    else:
        print(Fore.GREEN + "Final Residual" + Fore.RESET)
        print(residual_func(x_data, y_data, k))
        print(Fore.GREEN + "^" + Fore.RESET)
        print(Fore.GREEN + "Final Residual" + Fore.RESET)
        print(np.linalg.norm(residual_func(x_data, y_data, k)))
        print(Fore.GREEN + "^" + Fore.RESET)
        print(Fore.BLUE + "Parameters" + Fore.RESET)
        print(k)
        print(Fore.BLUE + "^" + Fore.RESET)
        print("Parameters did not converge")
        print(f"Last λ: {lam}")
        print(f"Last step_size: {step_size}")
    
    return k, path