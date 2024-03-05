import sympy as sp


if __name__ == "__main__":
    Ts, g, m, Jx, Jy, Jz, t= sp.symbols('Ts g m Jx Jy Jz t')
    
    # Zero-order hold Discretization of drone state space model
    A = sp.Matrix([[0,0,0,1,0,0,0,0,0,0,0,0],
         [0,0,0,0,1,0,0,0,0,0,0,0],
         [0,0,0,0,0,1,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,g,0,0,0,0],
         [0,0,0,0,0,0,-g,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,0,0,0,0,1],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0]])
    
    A_outer_cascade = sp.Matrix([[0,0,0,1,0,0,0,0,0],
         [0,0,0,0,1,0,0,0,0],
         [0,0,0,0,0,1,0,0,0],
         [0,0,0,0,0,0,0,g,0],
         [0,0,0,0,0,0,-g,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0]])
    
    A_inner_cascade = sp.Matrix([[0,0,0],
         [0,0,0],
         [0,0,0]])
    
    B = sp.Matrix([[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [1/m,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,1/Jx,0,0],
         [0,0,1/Jy,0],
         [0,0,0,1/Jz]])
    
    B_outer_cascade = sp.Matrix([[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [1/m,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]])
        
    B_inner_cascade = sp.Matrix([[1/Jx,0,0],
         [0,1/Jy,0],
         [0,0,1/Jz]])
    
    A_discrete = sp.exp(A * Ts)
    A_outer_cascade_discrete = sp.exp(A_outer_cascade * Ts)
    A_inner_cascade_discrete = sp.exp(A_inner_cascade * Ts)
    
    f = sp.exp(A * t)
    f_outer_cascade = sp.exp(A_outer_cascade * t)
    f_inner_cascade = sp.exp(A_inner_cascade * t)

    # Define the lower and upper bounds
    lower_bound = 0
    upper_bound = Ts

    # Calculate the definite integral with specified bounds
    integral = sp.integrate(f, (t, lower_bound, upper_bound))
    integral_outer_cascade = sp.integrate(f_outer_cascade, (t, lower_bound, upper_bound))
    integral_inner_cascade = sp.integrate(f_inner_cascade, (t, lower_bound, upper_bound))
    
    B_discrete = integral * B
    B_outer_cascade_discrete = integral_outer_cascade * B_outer_cascade
    B_inner_cascade_discrete = integral_inner_cascade * B_inner_cascade

    print("A_discrete:", A_discrete)
    print("integral: ", integral)
    print("B_discrete: ", B_discrete)