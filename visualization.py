import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import tabulate as tab
from colorama import Fore

def plot_guess(x_data, y_data, k_opt, message):
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Data')
    T = np.linspace(x_data.min(), x_data.max(), 100)
    Y = k_opt[0]*np.exp(-k_opt[1]*x_data) + k_opt[2]*np.exp(-k_opt[3]*x_data)
    plt.plot(T, Y, 'b-', label='Fit')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title(message)
    print("Optimized parameters:", k_opt)
    
def plot(func, path, variables):
    if len(variables) > 2:
        raise ValueError(f"Number of variables is greater than 2 (len(variables) = {len(variables)}), cannot plot 3D figure")
    if len(variables) == 1:
        raise NotImplementedError(f"Plotting for single variable functions is not yet implemented")
    buffer = 1
    grid = [100, 100]
    path_x = []
    path_y = []
    for pt in path:
        path_x.append(pt[0])
        path_y.append(pt[1])
    x_range = [float(min(path_x)) - buffer, float(max(path_x)) + buffer]
    y_range = [float(min(path_y)) - buffer, float(max(path_y)) + buffer]
    x1 = np.linspace(x_range[0], x_range[1], grid[0])
    x2 = np.linspace(y_range[0], y_range[1], grid[1])
    z = np.zeros((len(x1), len(x2)))
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            current_x = sp.Matrix([x1[i], x2[j]])
            z[j, i] = func(current_x)

    # draw contours
    contours=plt.contour(x1, x2, z, 100, cmap=plt.cm.gnuplot)
    
    # label the contour with numbers
    plt.clabel(contours, inline=1, fontsize=10)
    
    # label the axes
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    
    # plot the trajectory
    plt.plot(path_x, path_y, "o-", ms=5.5)
    
    # show
    plt.show()


def data_table(function, extremum, variables, path):
    # visualize as data table
    if extremum is not None:
        extremum = sp.Matrix(extremum)
        print("Extremum found at:", extremum.applyfunc(lambda x: round(x, ndigits=6)))
        print("Value of function at extremum: ", round(function(extremum), ndigits=6))
    print(Fore.RED + "Iterations" + Fore.RESET)
    data_list = []
    for i in range(len(path)):
        data = []
        data.append(i+1)
        for j in range(len(variables)):
            data.append(round(path[i][j], ndigits=6))
        data.append(round(function(path[i]), ndigits=6))
        data_list.append(data)
    
    variable_list = ["Iteration"]
    for i in range(len(variables)):
        variable_list.append(variables[i])
    variable_list.append('f(x)')
    if len(data_list) > 35:   
        print(tab.tabulate([variable_list] + data_list[:30] + [len(data_list[0]) * ["⋮"]] + data_list[-5:-1], tablefmt="simple", numalign="left", stralign="center", headers="firstrow"))
    else:
        print(tab.tabulate([variable_list] + data_list, tablefmt="simple", numalign="left", stralign="center", headers="firstrow"))
    print(Fore.RED + "^" + Fore.RESET)