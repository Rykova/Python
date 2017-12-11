import TasksOfNBodiesFunction
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import Task11

from matplotlib.legend_handler import HandlerLine2D

if __name__ == '__main__':
    MODES = [
                ("scipy",1), ("verlet",2),
                ("verlet-threading",3), ("verlet-multiprocessing",4),
                ("verlet-cython without typed memoryview",5),("verlet-cython with typed memoryview",6), ("verlet-openmp without typed memoryview",7),
                ("verlet-openmp with typed memoryview",8),("verlet-opencl",9)
            ]
    # Calculate_Defect()
    # Get_Average_Time(MODES)
    # Time_of_all_methods(MODES)
    fig = plt.figure(figsize=(20, 6))

    list_of_K = [10, 50, 100, 200, 500, 1000]

    time_scipy = TasksOfNBodiesFunction.GetTimeSlow(list_of_K)
    line1, = plt.plot(list_of_K, time_scipy, 'g', linewidth=1, label='verle')

    time_values_3 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K,"verlet-threading")
    line2, = plt.plot(list_of_K, time_values_3, 'y', linewidth=1, label="verlet-threading")

    time_values_4 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-multiprocessing")
    line3, = plt.plot(list_of_K, time_values_4, 'b', linewidth=1, label="verlet-multiprocessing")

    time_values_5 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-cython without typed memoryview")
    line4, = plt.plot(list_of_K, time_values_5, 'k', linewidth=1, label="verlet-cython without typed memoryview")

    time_values_6 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-cython with typed memoryview")
    line5, = plt.plot(list_of_K, time_values_6, 'r', linewidth=1, label="verlet-cython with typed memoryview")

    time_values_7 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-openmp without typed memoryview")
    line6, = plt.plot(list_of_K, time_values_7, 'chocolate', linewidth=1, label="verlet-openmp without typed memoryview")

    time_values_8 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-openmp with typed memoryview")
    line7, = plt.plot(list_of_K, time_values_8, 'm', linewidth=1, label="verlet-openmp with typed memoryview")

    time_values_9 = TasksOfNBodiesFunction.GetTimeForKBodies(list_of_K, "verlet-opencl")
    line8, = plt.plot(list_of_K, time_values_9, 'c', linewidth=1, label="verlet-opencl")

    plt.xlabel('Число точек')
    plt.ylabel('Время')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    savefig('results.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(20, 6))
    line1, = plt.plot(list_of_K,  np.divide(time_scipy,time_scipy), 'g', linewidth=1, label='verle')

    line2, = plt.plot(list_of_K, np.divide(time_scipy,time_values_3), 'y', linewidth=1, label="verlet-threading")


    line3, = plt.plot(list_of_K, np.divide(time_scipy,time_values_4), 'b', linewidth=1, label="verlet-multiprocessing")


    line4, = plt.plot(list_of_K, np.divide(time_scipy,time_values_5), 'k', linewidth=1, label="verlet-cython without typed memoryview")


    line5, = plt.plot(list_of_K, np.divide(time_scipy,time_values_6), 'r', linewidth=1, label="verlet-cython with typed memoryview")


    line6, = plt.plot(list_of_K, np.divide(time_scipy,time_values_7), 'chocolate', linewidth=1,
                      label="verlet-openmp without typed memoryview")


    line7, = plt.plot(list_of_K, np.divide(time_scipy,time_values_8), 'm', linewidth=1, label="verlet-openmp with typed memoryview")


    line8, = plt.plot(list_of_K, np.divide(time_scipy,time_values_9), 'c', linewidth=1, label="verlet-opencl")

    plt.xlabel('Число точек')
    plt.ylabel('Ускорение')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    savefig('results2.pdf', bbox_inches='tight')

    #ax = fig.add_subplot(211)
    #for method in MODES:
    #    line1, = plt.plot(list_of_K, time_values, 'g', linewidth=1, label=method[0])
    #plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    #plt.plot(list_of_K, time_values, color="green")


    #ax1 = plt.subplot(212, sharex=ax)
    #plt.plot(list_of_K, boost_values, color="red")
    #plt.xlabel('Число точек')
    #plt.ylabel('Ускорение')
    #plt.show()

    #savefig('results.pdf', bbox_inches='tight')
