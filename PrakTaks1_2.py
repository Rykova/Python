
from sympy import Symbol, solve, lambdify, Matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


km1 = Symbol('km1', Positive=True)
km2 = Symbol('km2', Positive=True)
k3_0 = Symbol('k3_0', Positive=True)
alpha =Symbol('alpha', Positive=True)
k1 = Symbol('k1', Positive=True)
k2 = Symbol('k2', Positive=True)
x = Symbol("x", Positive=True)
y = Symbol("y", Positive=True)

alphaval = 16.0
km1val = 0.01
km2val = 0.01
k3_0val = 10.0

alpha_range = [10,15,18,20,25]
k3_range = [1,5,10,50,100]

eq1 = k1 * (1 - x - y)- km1*x - x * y * k3_0*(1-y)**alpha
eq2 = k2 * (1 - x - y) ** 2 - km2 * y ** 2 - x * y * k3_0*(1-y)**alpha


solution = solve([eq1, eq2], x, k2)
xSolution = solution[0][0]
k2Solution = solution[0][1]


A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jacA = A.jacobian(var_vector)
detA = jacA.det()
traceA = jacA.trace()

Y = np.arange(0, 1, 1e-3)



def AnalysisK1K2():
    # Neutral lines
    print("Step one")
    k2TraceSol = solve(traceA.subs(x,xSolution),k2)[0]
    k1JointTraceSol = solve(k2TraceSol - k2Solution,k1)[0]
    k2JointTraceSol = k2Solution.subs(k1,k1JointTraceSol)
    k1Trace_y = lambdify((y,km1,km2,k3_0,alpha),k1JointTraceSol,'numpy')
    k2Trace_y = lambdify((y,km1,km2,k3_0,alpha),k2JointTraceSol,'numpy')

    print("Step two")
    # Multiplicity lines
    k2DetSolution = solve(detA.subs(x, xSolution), k2)

    eqprek1 = k2Solution - k2DetSolution[0]
    print(eqprek1)
    k1JointDetSolution = solve(eqprek1.subs(k3_0, k3_0val ).subs(km1, km1val).subs(km2, km2val).subs(alpha,alphaval), k1)
    print(k1JointDetSolution)
    k2JointDetSolution = k2Solution.subs(k1, k1JointDetSolution[0])
    print(k2JointDetSolution)
    k1Det_of_y = lambdify((y, km1, km2, k3_0, alpha), k1JointDetSolution[0],'numpy')
    k2Det_of_y = lambdify((y, km1, km2, k3_0, alpha), k2JointDetSolution[0],'numpy')

    print("Step three")
   # YY = Y[k2Det_of_y(Y, km1val,km2val,k3_0val,alphaval) > 0]

    print("Step six")
    plt.plot(k2Trace_y(Y, km1val,km2val,k3_0val,alphaval), k1Trace_y(Y, km1val,km2val,k3_0val,alphaval),linestyle='--', linewidth=1.5, label='neutral')
    plt.plot(k2Det_of_y(Y, km1val,km2val,k3_0val,alphaval), k1Det_of_y(Y, km1val,km2val,k3_0val,alphaval),
             linewidth=.8, label='multiplicity')
    plt.xlabel(r'$k_1$')
    plt.ylabel(r'$k_2$')
    plt.xlim([-0.0, 0.2])
    plt.ylim([-0.0, 0.2])
    plt.legend(loc=0)
    plt.show()
    return
def solveSystem(init, k1val, k1mval, k2val, k2mval, k3_0val,alphaval, dt, iterations):

    f1 = lambdify((x, y, k1, km1, k3_0,alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0,alpha), eq2)

    def rhs(xy, times):
        return [f1(xy[0], xy[1], k1val, k1mval, k3_0val,alphaval), f2(xy[0], xy[1],k2val, k2mval, k3_0val,alphaval)]

    times = np.arange(iterations) * dt
    return odeint(rhs, init, times), times
def autocol():
    res, times = solveSystem([0.38, 0.22], 0.03, 0.01, 0.05, 0.01, 10, 16, 1e-2, 1e6)

    ax = plt.subplot(211)
    plt.plot(times, res[:, 0])
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel('x')
    plt.grid()
    ax1 = plt.subplot(212, sharex=ax)
    plt.plot(times, res[:, 1], color='red')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.show()
    return
def streamplot(k1val, k1mval, k2val, k2mval, k3_0val,alphaval):
    f1 = lambdify((x, y, k1, km1, k3_0, alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0, alpha), eq2)
    Y, X = np.mgrid[0:.5:1000j, 0:1:2000j]
    U = f1(X, Y, k1val, k1mval, k3_0val,alphaval)
    V = f2(X, Y, k2val, k2mval, k3_0val,alphaval)
    velocity = np.sqrt(U*U + V*V)
    plt.streamplot(X, Y, U, V, density = [2.5, 0.8], color=velocity)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#streamplot(0.03, 0.01, 0.05, 0.01, 10,16)
#AnalysisK1K2()
autocol()
#AnalysisK1K2()
#AnalysisK2(0.03,0.01,0.01,10,16,1e-12)