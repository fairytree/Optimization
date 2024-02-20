# import the required packages
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def visualize(func, x_range: list[float, float], y_range: list[float, float], grid: list[int, int], x_1: list[float], x_2: list[float], number_labels: bool = False):
    # define the variables x and y
    x, y = sp.symbols("x y")
    
    
    # values of func at certain points on a grid
    x1 = np.linspace(x_range[0], x_range[1], grid[0])
    x2 = np.linspace(y_range[0], y_range[1], grid[1])
    z = np.zeros((len(x1), len(x2)))
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            z[j, i] = func.evalf(subs={"x": x1[i], "y": x2[j]})

    # draw contours
    contours=plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
    
    # label the contour with numbers if number_labels == True
    if number_labels:
        plt.clabel(contours, inline=1, fontsize=10)

    # label the axes
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    
    # plot the trajectory
    plt.plot(x_1, x_2, "o-", ms=5.5)
    
    # show
    plt.show()
