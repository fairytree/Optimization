# import the required packages
import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as au
from autograd import grad, jacobian
import scipy

def func(x): # Objective function (Branin function)
    return (x[1] - (5.1/(4*au.pi**2))*x[0]**2 + (5/au.pi)*x[0] - 6)**2 + 10*(1 - 1/(8*au.pi))*au.cos(x[0]) + 10
    
Df = grad(func) # Gradient of the objective function

from scipy.optimize import line_search
NORM = np.linalg.norm

x1 = np.linspace(-5, 16, 100)
x2 = np.linspace(-5, 16, 100)
z = np.zeros(([len(x1), len(x2)]))
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        z[j, i] = func([x1[i], x2[j]])

contours=plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
plt.clabel(contours, inline=1, fontsize=10)

plt.xlabel("$x_1$ ->")

plt.ylabel("$x_2$ ->")

def rank_1(Xj, tol, alpha_1, alpha_2):
    x1 = [Xj[0]]
    x2 = [Xj[1]]
    Bf = np.eye(len(Xj))
    
    while True:
        Grad = Df(Xj)
        delta = -Bf.dot(Grad) # Selection of the direction of the steepest descent
        
        
        start_point = Xj # Start point for step length selection 
        beta = line_search(f=func, myfprime=Df, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] # Selecting the step length
        if beta!=None:
            X = Xj+ beta*delta
        if NORM(Df(X)) < tol:
            x1 += [X[0], ]
            x2 += [X[1], ]
            plt.plot(x1, x2, "rx-", ms=5.5) # Plot the final collected data showing the trajectory of optimization
            plt.show()
            return X, func(X)
        else:
            Dj = X - Xj # See line 17 of the algorithm
            Gj = Df(X) - Grad # See line 18 of the algorithm
            w = Dj - Bf.dot(Gj) # See line 19 of the algorithm
            wT = w.T # Transpose of w
            sigma = 1/(wT.dot(Gj)) # See line 20 of the algorithm
            W = np.outer(w, w) # Outer product between w and the transpose of w
            Delta = sigma*W # See line 21 of the algorithm
            if abs(wT.dot(Gj)) >= 10**-8*NORM(Gj)*NORM(w): # update criterion (See line 22-24)
                Bf += Delta          
            Xj = X # Update to the new iterate
            x1 += [Xj[0], ]
            x2 += [Xj[1], ]
            
rank_1(np.array([11.8, 5.75]), 10**-5, 10**-4, 0.24)

