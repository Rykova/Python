import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import sympy
from sympy import symbols
from sympy import *

#однопараметрический анализ по параметру k2
alpha = 16
k1 = 0.03
km1 = 0.01
km2 = 0.01
k3_0 = 5

tol = 1e-12
def AnalysisK2(k1, km1,km2,k3_0,alpha,tol):
    y = np.linspace(0.001, 0.987, 1000)
    N = np.size(y)

    x = np.zeros(N)
    z = np.zeros(N)
    phi = np.zeros(N)
    phi_m = np.zeros(N)
    k2 = np.zeros(N)
    sp = np.zeros(N)
    delA = np.zeros(N)
    DI = np.zeros(N)
    sn_x = []
    sn_y = []

    phi[0] = pow((1 - y[0]), alpha)
    phi_m[0] = pow((1 - y[0]), alpha - 1)
    x[0] = k1 * (1 - y[0]) / (k1 + km1 + k3_0 * phi[0] * y[0])
    z[0] = 1 - x[0] - y[0]
    k2[0] = km2 * pow(y[0], 2) * pow((k1 + km1 + k3_0 * phi[0] * y[0]), 2) + (k1 + km1 + k3_0 * phi[0] * y[
        0]) * k1 * k3_0 * phi[0] * y[0] * (1 - y[0])
    k2[0] = k2[0] / (pow((1 - y[0]), 2) * pow((km1 + k3_0 * phi[0] * y[0]), 2))

    a11 = -k1 - km1 - k3_0 * phi[0] * y[0]
    a12 = -k1 - k3_0 * x[0] + k3_0 * alpha * phi_m[0] * y[0] * x[0]
    a21 = -2 * k2[0] * z[0] - k3_0 * phi[0] * y[0]
    a22 = -2 * k2[0] * z[0] - 2 * km2 * y[0] - k3_0 * phi[0] * x[0] + k3_0 * alpha * phi_m[0] * y[0] * x[0]

    sp[0] = a11 + a22
    delA[0] = a11 * a22 - a12 * a21
    DI[0] = pow(sp[0], 2) - 4 * delA[0]

    for i in range(1, N):
        phi[i] = pow((1 - y[i]), alpha)
        phi_m[i] = pow((1 - y[i]), alpha - 1)
        x[i] = k1 * (1 - y[i]) / (k1 + km1 + k3_0 * phi[i] * y[i])
        z[i] = 1 - x[i] - y[i]
        k2[i] = km2 * pow(y[i], 2) * pow((k1 + km1 + k3_0 * phi[i] * y[i]), 2) + (k1 + km1 + k3_0 * phi[i] * y[
            i]) * k1 * k3_0 * phi[i] * y[i] * (1 - y[i])
        k2[i] = k2[i] / (pow((1 - y[i]), 2) * pow((km1 + k3_0 * phi[i] * y[i]), 2))

        a11 = -k1 - km1 - k3_0 * phi[i] * y[i]
        a12 = -k1 - k3_0 * phi[i] * x[i] + k3_0 * alpha * phi_m[i] * y[i] * x[i]
        a21 = -2 * k2[i] * z[i] - k3_0 * phi[i] * y[i]
        a22 = -2 * k2[i] * z[i] - 2 * km2 * y[i] - k3_0 * phi[i] * x[i] + k3_0 * alpha * phi_m[i] * y[i] * x[i]

        sp[i] = a11 + a22
        delA[i] = a11 * a22 - a12 * a21
        DI[i] = pow(sp[i], 2) - 4 * delA[i]

        if (delA[i] * delA[i - 1] < tol):
            y_new_point = y[i - 1] - delA[i - 1] * (y[i] - y[i - 1]) / (delA[i] - delA[i - 1])
            k2_new_point = km2 * pow(y_new_point, 2) * pow((k1 + km1 + k3_0 * (1 - y_new_point) ** alpha * y_new_point),
                                                           2) + \
                           (k1 + km1 + k3_0 * (1 - y_new_point) ** alpha * y_new_point) * k1 * k3_0 * (
                                                                                                      1 - y_new_point) ** alpha * y_new_point * (
                           1 - y_new_point)
            k2_new_point = k2_new_point / (
            pow((1 - y_new_point), 2) * pow((km1 + k3_0 * (1 - y_new_point) ** alpha * y_new_point), 2))
            x_new_point = k1 * (1 - y_new_point) / (k1 + km1 + k3_0 * (1 - y_new_point) ** alpha * y_new_point)
            sn_x.append([k2_new_point,x_new_point])
            sn_y.append([k2_new_point,y_new_point])
            plt.plot(k2_new_point, x_new_point, 'k*', marker='o', label="node")
            plt.plot(k2_new_point, y_new_point, 'r', marker='o', label="node")

    plt.title('Однопараметрический анализ. Завсимость стационарных решений от параметра k2')
    line1, = plt.plot(k2, x, 'b--', label="x")
    line2, = plt.plot(k2, y, 'k', label="y")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    plt.xlim((0, 0.7))
    plt.xlabel('k2')
    plt.ylabel('x,y')
    plt.grid(True)
    plt.show()
    return sn_x, sn_y


alpha_range = [10,15,18,20,25]
k3_range = [1,5,10,50,100]

for elem in k3_range:
   # AnalysisK2(k1,km1,km2,k3_0,elem,tol)
   AnalysisK2(k1, km1, km2, elem, alpha, tol)
