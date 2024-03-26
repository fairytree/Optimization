import numpy as np
import matplotlib.pyplot as plt

def gauss_newton_marquardt(x_init, y_data, residual_func, jacobian_func, lam = 0.0001, max_iter_outer=1000, max_iter_inner=100, tol=1e-8):
    x = x_init
    for i in range(max_iter_outer):
        residual = residual_func(x)
        J = jacobian_func(x)
        
        # Calculate the Gauss-Newton step with Marquardt modification: (J.T * J + lambda * I) * delta = - J.T * residual
        matrix_A = J.T.dot(J) + lam * np.eye((J.T).shape[0])
        vector_b = -J.T.dot(residual)
        delta = np.linalg.lstsq(matrix_A, vector_b, rcond=None)[0]
        
        # Use bisection rule to obtain step size lambda
        step = 1
        for j in range(max_iter_inner):
            # Update the parameters
            x_new = x + step * delta
            # Calculate the norm of residual (cost function)
            ssr_old = np.linalg.norm(residual)
            ssr_new = np.linalg.norm(residual_func(x_new))
            if ssr_new < ssr_old:
                x_old = x
                x = x_new            
                break
            else:
                step /= 2
    
        # Check convergence
        if np.mean(np.abs(delta/x_old)) < tol:
            print(f"Converged in {i+1} iterations.")
            break
    else:
        print("Did not converge.")
    
    return x

# Define the residual function
def residual_func(x):
    return y_data - (x[0]*np.exp(-x[1]*t_data) + x[2]*np.exp(-x[3]*t_data))

# Define the Jacobian function
def jacobian_func(x):
    J = np.zeros((len(y_data), len(x)))
    J[:, 0] = -np.exp(-x[1]*t_data)
    J[:, 1] = x[0]*t_data*np.exp(-x[1]*t_data)
    J[:, 2] = -np.exp(-x[3]*t_data)
    J[:, 3] = x[2]*t_data*np.exp(-x[3]*t_data)
    return J

def plot_guess(x, message):
    plt.plot(t_data, y_data, 'ro', markersize=8, label='Data')
    T = np.linspace(t_data.min(), t_data.max(), 100)
    Y = x[0]*np.exp(-x[1]*t_data) + x[2]*np.exp(-x[3]*t_data)
    plt.plot(T, Y, 'b-', label='Fit')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title(message)
    

if __name__ == "__main__":
    # Example
    t_data = np.linspace(-10, 10, 100)
    y_true = 2.0 * np.exp(-0.5 * t_data) + 1.5 * np.exp(-0.2 * t_data)
    np.random.seed(0)
    y_data = y_true + np.random.normal(0, 0.01, size=t_data.shape)
    x_init = np.array([-1.5, 0.8, 1.2, 0.3])
    x_opt = gauss_newton_marquardt(x_init, y_data, residual_func, jacobian_func)

    print("Optimized parameters:", x_opt)
    plt.figure()
    plot_guess(x_init, "initial guess")
    plt.figure()
    plot_guess(x_opt, "last Iteration")
    plt.show()