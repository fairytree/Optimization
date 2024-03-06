import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import tabulate as tab


def plot(func, path, variables):
    if len(variables) > 2:
        print("len(variables): ", len(variables), ", cannot plot 3D figure")
        return None
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
    # visualize data table
    extremum = sp.Matrix(extremum)
    print("Extremum found at:", extremum.applyfunc(lambda x: round(x, 6)))
    # print("Value of function at extremum: ", round(function(extremum),6))
    print("Each iteration is as follows: ")
    data_list = []
    for i in range(len(path)):
        data = []
        data.append(i+1)
        for j in range(len(variables)):
            data.append(round(path[i][j],6))
        data.append(round(function(path[i]),6))
        data_list.append(data)
    
    variable_list = ["Iteration"]
    for i in range(len(variables)):
        variable_list.append(variables[i])
    variable_list.append('f(x)')    
    print(tab.tabulate([variable_list] + data_list, tablefmt="simple_grid", numalign="center"))