from __future__ import absolute_import #включение абсолютных путей по умолчанию для импорта
from __future__ import print_function
import copy
import numpy as np
import numpy.linalg as nlg
import threading
import multiprocessing as mp
import time
import pyopencl as cl
import pyopencl.cltypes

import random

from scipy.integrate import odeint
import cythverlet

MassSun = 1.99 * pow(10, 30)
MassEarth = 5.98 * pow(10, 24)
MassMoon = 7.32 * pow(10, 22)
MassMerc = 3.285 *pow(10,23)
M = 500
T = 5*12 * 2592000
r_norm = 1.496 * 10 ** 11


def TaskOfNbodiesVerle(type):

    list_of_mass_all = [MassMoon/MassSun, MassEarth/MassSun, 1, MassMerc/MassSun]
    list_of_radius_and_velocity_all = np.zeros((4, 6))


    #first body moon
    list_of_radius_velocity = [0, 1.496*10**11/r_norm + 384467000/r_norm, 0, 1022/r_norm +29.783*10 **3/r_norm, 0, 0]
    list_of_radius_and_velocity_all[0,:] = list_of_radius_velocity

    #second body earth
    list_of_radius_velocity = [0, 1.496*10**11/r_norm, 0, 29.783*10 **3/r_norm, 0, 0]
    list_of_radius_and_velocity_all[1, :] = list_of_radius_velocity

    # third sun
    list_of_radius_velocity = [0, 0, 0, 0, 0, 0]
    list_of_radius_and_velocity_all[2, :] = list_of_radius_velocity

    # fourth mercury
    list_of_radius_velocity = [0, 57910000*1000/r_norm, 0, 47.36*1000/r_norm, 0, 0]
    list_of_radius_and_velocity_all[3, :] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init =list_of_radius_and_velocity_all.reshape((6 * N))


    if(type == "verlet"):
        result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return  result
    if (type =="scipy"):
        time_span = np.linspace(0,T,M)
        result = odeint(g,init,time_span,args=(list_of_mass_all,N))
        result2 = result.reshape((M, N, 6))
        return result2
    if (type == "verlet-threading"):
        result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result
    if (type == "verlet-cython without typed memoryview"):
        result = cythverlet.cverletnotypedmemoryview(np.asarray(list_of_radius_and_velocity_all), np.asarray(list_of_mass_all), M, T)
        return np.asarray(result)
    if (type == "verlet-cython with typed memoryview"):
        result = cythverlet.cverlettypedmemoryview(np.asarray(list_of_radius_and_velocity_all), np.asarray(list_of_mass_all), M, T)
        return np.asarray(result)
    if (type == "verlet-openmp without typed memoryview"):
        result = cythverlet.cverlet_openmp(np.asarray(list_of_radius_and_velocity_all),
                                                   np.asarray(list_of_mass_all), M, T)
        return np.asarray(result)
    if (type == "verlet-openmp with typed memoryview"):
        result = cythverlet.cverlettypedmemoryview_openmp(np.asarray(list_of_radius_and_velocity_all),
                                                    np.asarray(list_of_mass_all), M, T)
        return np.asarray(result)
    if (type == "verlet-multiprocessing"):
        result = VerletMethodMultiprocessing(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result
    if (type == "verlet-opencl"):
        result = Verlet_OpenCl(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result

def solveForOneBody(q, q_out, list_of_radius_and_velocity,body,shared_pos,tau,list_of_mass_all,M, events1, events2):
        result = np.zeros((M,6))
        N = len(list_of_radius_and_velocity)
        a = np.zeros((N, 3))
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, body)
        for j in range(1, M):
            list_of_radius_and_velocity_new = np.zeros((N,6))
            Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[body], tau, body)

            q.put([body, list_of_radius_and_velocity_new[body,0:3]])
            events1[body].set()
            if (body == 0):
                for i in range(N):
                    events1[i].wait()
                    events1[i].clear()
                for i in range(0, N):
                    tmp = q.get()
                    shared_pos[tmp[0]*6:tmp[0]*6+3] = tmp[1]

                for i in range(0,N):
                    events2[i].set()
            else:
                events2[body].wait()
                events2[body].clear()

            arr = np.frombuffer(shared_pos.get_obj())
            list_of_radius_and_velocity_new = arr.reshape((N, 6))
            a_new = np.zeros((N,3))
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, body)
            Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[body], a_new[body], tau, body)
            list_of_radius_and_velocity[body] = copy.copy(list_of_radius_and_velocity_new[body])
            a = copy.copy(a_new)
            result[j] = list_of_radius_and_velocity[body]

        q_out.put([body, result])

def VerletMethodMultiprocessing(list_of_radius_and_velocity, list_of_mass_all, M, T):

    tau = T / M
    N = len(list_of_radius_and_velocity)
    pos = np.asarray(list_of_radius_and_velocity)
    shared_pos = mp.Array('d',np.zeros(6*N))

    if __name__ == 'TasksOfNBodiesFunction':
        result = np.zeros((M, N, 6))

        events1 = []
        events2 = []

        for elem in list_of_mass_all:
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[-1].clear()
            events2[-1].clear()

        q = mp.Queue()
        q_out = mp.Queue()
        processes = []
        for body in range(0, N):
            processes.append(mp.Process(target=solveForOneBody,
                           args=(q, q_out, list_of_radius_and_velocity, body,shared_pos, tau, list_of_mass_all,M, events1, events2)))

            processes[-1].start()


        for i in range(0, N):
            tmp = q_out.get()
            result[:,tmp[0],:] = tmp[1]

        result[0] = copy.copy(list_of_radius_and_velocity)

        for process in processes:
            process.join()


        return result

def solveForOneBodyKbodies(q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all, n1, n2,M, events1,
                        events2):
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    a = np.zeros((N, 3))

    for i in range(n1, n2):
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, i)
    for j in range(1, M):
        list_of_radius_and_velocity_new = np.zeros((N, 6))


        for i in range(n1, n2):
            Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[i], tau, i)


        for i in range(n1, n2):
            q.put([i, list_of_radius_and_velocity_new[i, 0:3]])

        events1[body].set()

        if (body == 0):
            for i in range(0,4):
                events1[i].wait()
                events1[i].clear()


            for i in range(0,N):
                tmp = q.get()
                shared_pos[tmp[0] * 6:tmp[0] * 6 + 3] = tmp[1]
            for i in range(0, 4):
                events2[i].set()
        else:
            events2[body].wait()
            events2[body].clear()


        arr = np.frombuffer(shared_pos.get_obj())
        list_of_radius_and_velocity_new = arr.reshape((N, 6))
        a_new = np.zeros((N, 3))

        for i in range(n1, n2):
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, i)

        for i in range(n1, n2):
            Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[i], a_new[i],
                                    tau, i)

        for i in range(n1, n2):
            list_of_radius_and_velocity[i] = copy.copy(list_of_radius_and_velocity_new[i])
        a = copy.copy(a_new)



        for i in range(n1, n2):
            result[j, ] = list_of_radius_and_velocity[i]



    for i in range(n1, n2):
        q_out.put([i, result[:, i, :]])


def VerletMethodMultiprocessingKbodies(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau = T / M
    N = len(list_of_radius_and_velocity)
    pos = np.asarray(list_of_radius_and_velocity)
    shared_pos = mp.Array('d', np.zeros(6 * N))

    if __name__ == 'TasksOfNBodiesFunction':
        result = np.zeros((M, N, 6))

        events1 = []
        events2 = []

        for elem in range(0, 4):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[-1].clear()
            events2[-1].clear()

        q = mp.Queue()
        q_out = mp.Queue()
        processes = []

        for body in range(0, 4):
            processes.append(mp.Process(target=solveForOneBodyKbodies,
                                        args=(
                                        q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all,
                                        int(body * N / 4), int(body * N / 4 + N / 4),M, events1, events2)))

            processes[-1].start()

        for i in range(0, N):
            tmp = q_out.get()
            result[:, tmp[0], :] = tmp[1]

        result[0] = copy.copy(list_of_radius_and_velocity)

        for process in processes:
            process.join()


        return result

def Velocity_form_for_v(list,list_new,a,a_new,tau,j):
    list_new[j,3:6]=list[j,3:6]+0.5*(a+a_new)*tau

def Velocity_form_for_x(list,list_new,a,tau,j):
    list_new[j,0:3]=list[j,0:3]+list[j,3:6]*tau+0.5*a*tau**2

def accelaration_for_i_body(a, N, list_of_data,list_of_mass,i):
    G=6.67 * 10 **(-11)*MassSun/pow(r_norm,3) #gravitation const

    for j in range(0,N):
        if i!=j:
            a[i]+=G*list_of_mass[j]*(list_of_data[j,0:3]-list_of_data[i,0:3])/nlg.norm(list_of_data[j,0:3]-list_of_data[i,0:3],2) ** 3


def VerletMethod(list_of_radius_and_velocity, list_of_mass_all, N, M, T):
    tau=T/M
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    #acceleration
    a = np.zeros((N, 3))
    for i in range(0, N):
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, i)

    for i in range(1,M):

        list_of_radius_and_velocity_new = np.zeros((N,6))
        for j in range(0,N):
            Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)


        a_new = np.zeros((N, 3))

        for k in range(0, N):
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, k)
        for j in range(0,N):
            Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)
        list_of_radius_and_velocity=copy.copy(list_of_radius_and_velocity_new)
        a=copy.copy(a_new)
        result[i]=copy.copy(list_of_radius_and_velocity)


    return result


def ThreadingWork(M,N,list_of_thread,th_ev):
    for i in range(1,M):
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()


def ThreadMethod(result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev):

    N=len(list_of_radius_and_velocity)
    a = np.zeros((N, 3))
    accelaration_for_i_body(a, N, list_of_radius_and_velocity,list_of_mass_all,j)
    for i in range(1,M):
        list_of_radius_and_velocity_new[j]=np.zeros(6)
        Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()
        a_new = np.zeros((N, 3))
        accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, j)
        Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)
        list_of_radius_and_velocity[j]=copy.copy(list_of_radius_and_velocity_new[j])
        a=copy.copy(a_new)
        result[i,j]=copy.copy(list_of_radius_and_velocity[j])
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()

def VerletMethodThreading(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau=T/M
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    th_ev = threading.Event()
    list_of_radius_and_velocity_new = np.zeros((N, 6))
    list_of_thread = []
    for j in range(0, N):
        th = threading.Event()
        list_of_thread.append(th)
    Threads = threading.Thread(target=ThreadingWork, name="ThreadingWork", args=(M, N, list_of_thread, th_ev))
    Threads.start()
    for j in range(0, N):
        t = threading.Thread(target=ThreadMethod, name="thread" + str(j), args=(result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev))
        t.start()
    Threads.join()

    return result

def g(list_of_data,time_span,list_of_mass,N):
    G = 6.67 * 10 ** (-11)*MassSun/pow(r_norm,3)
    mass_of_funct = np.zeros(6*N)
    for i in range(0, N):
        f1 = list_of_data[6 * i + 3 : 6 * i + 6]
        f2 = np.zeros(3)
        for j in range(0, N):
            if (i != j):
                f2 += G * list_of_mass[j] * (list_of_data[6*j:6*j+3] - list_of_data[6*i:6*i+3]) / nlg.norm(list_of_data[6*j:6*j+3] - list_of_data[6*i:6*i+3], 2) ** 3
        mass_of_funct[6*i:6*i+3] = f1
        mass_of_funct[6*i+3:6*i+6] = f2
    return mass_of_funct

    # расчёт погрешности

#Реализовать расчет погрешности произвольного солвера по сравнению с
#выдаваемым scipy.odeint
def Calculate_Defect():
    result_odeint = TaskOfNbodiesVerle("scipy")
    result_solver = TaskOfNbodiesVerle("verlet-threading")
    defect = np.max(np.max(np.max(result_solver - result_odeint)))
    print("Погрешность = " + repr(defect))

# среднее время расчета задачи N тел для заданной функции расчета по M вычислениям
def average_time(M_iter, method):
    av_time = 0
    for i in range(0, M_iter):
        t = time.time()
        result = TaskOfNbodiesVerle(method)
        av_time += time.time() - t
    return av_time / M_iter

def Get_Average_Time(MODES):
    M_iter = 3
    for elem in MODES:
        if (elem[0] == "verlet-multiprocessing"):
            continue
        print("Среднее время расчёта" + repr(elem[0]) + "= " + repr(average_time(M_iter, elem[0])))


#Определить самый быстрый и самый медленный методы и ускорение для
#всех методов по сравнению с самым медленным.
def Time_of_all_methods(MODES):
    min_method = "scipy"
    max_method = "scipy"
    t = time.time()
    result = TaskOfNbodiesVerle("scipy")
    min_time = time.time() - t
    max_time = time.time() - t
    list = []
    for elem in MODES:
        if (elem[0] == "scipy" or elem[0] == "verlet-multiprocessing"):
            continue
        t = time.time()
        result = TaskOfNbodiesVerle(elem[0])
        time_for_method = time.time() - t
        list.append((elem[0], time_for_method))
        if (time_for_method < min_time):
            min_time = time_for_method
            min_method = elem[0]
        if (time_for_method > max_time):
            max_time = time_for_method
            max_method = elem[0]
    print("Самый быстрый метод - " + repr(min_method) + ", время = " + repr(min_time))
    print("Самый медленный метод - " + repr(max_method) + ", время = " + repr(max_time))

    print("Ускорение:")
    for elem in list:
        if (elem[0] == "max_method"):
            continue
        print(repr(elem[0]) + " = " + repr(elem[1] / max_time))

#Реализовать функцию, генерирующую произвольный набор из K частиц c
#небольшими скоростями, достаточно большими массами и достаточно
#разнесенные в пространстве.
#Расчетное время: 10 * линейный размер расчетной области / максимальная
#скорость.

def TaskOfKRandomBodies(K,type):

    list_of_mass_all = np.zeros(K)
    list_of_radius_and_velocity_all = np.zeros((K, 6))
    step = random.randrange(10**3,10**10)

    for i in range(0,K):
        list_of_mass_all[i] = random.randrange(pow(10, 24),pow(10, 30))
        temp = random.randrange(200,500)

        list_of_radius_velocity = [0,i*step,0,
                                   temp,0,0]
        list_of_radius_and_velocity_all[i,:] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init = list_of_radius_and_velocity_all.reshape((6 * N))


    T = 500* step*10 #Расчетное время
    M = 100


    t = time.time()
    if (type == "verlet"):
        result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return result, time.time() - t
    if (type == "scipy"):
        time_span = np.linspace(0, T, M)
        result = odeint(g, init, time_span, args=(list_of_mass_all, N))
        result2 = result.reshape((M, N, 6))
        return result2, time.time() - t
    if (type == "verlet-threading"):
        result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result, time.time() - t
    if (type == "verlet-cython without typed memoryview"):
        result = cythverlet.cverletnotypedmemoryview(np.asarray(list_of_radius_and_velocity_all),
                                                     np.asarray(list_of_mass_all), M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-cython with typed memoryview"):
        result = cythverlet.cverlettypedmemoryview(np.asarray(list_of_radius_and_velocity_all),
                                                   np.asarray(list_of_mass_all), M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-openmp without typed memoryview"):
        result = cythverlet.cverlet_openmp(np.asarray(list_of_radius_and_velocity_all),
                                           np.asarray(list_of_mass_all), M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-openmp with typed memoryview"):
        result = cythverlet.cverlettypedmemoryview_openmp(np.asarray(list_of_radius_and_velocity_all),
                                                          np.asarray(list_of_mass_all), M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-multiprocessing"):
        result = VerletMethodMultiprocessingKbodies(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result,time.time() - t
    if (type == "verlet-opencl"):
        result = Verlet_OpenCl(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result,time.time() - t


#Провести расчет для K равного 10 50 100 200 500 1000.
def GetTimeSlow(list_of_K):
    time_scipy = []
    for K in list_of_K:
        print(repr('verlet') + " " + repr(K))
        result, time = TaskOfKRandomBodies(K, 'verlet')
        time_scipy.append(time)
    return time_scipy

def GetTimeForKBodies(list_of_K,method):
   #list_of_K = [10,50,100,200,500,1000]
    time_values = []
    i = 0
    for K in list_of_K:
        if(method == "verlet" or method == "scipy"):
            continue
        print(repr(method) + " " + repr(K))
        result,time = TaskOfKRandomBodies(K,method)
        time_values.append(time)
        i = i + 1

    return time_values

#opencl
def Verlet_OpenCl(list_of_radius_and_velocity,list_of_mass, M_body,T_body):

    N_body = len(list_of_radius_and_velocity)
    tau_body = T_body/M_body
    N = np.array(N_body)
    M = np.array(M_body)
    T = np.array(T_body)
    tau = np.array(tau_body)


    demo_r = np.empty((N, 5), dtype=np.uint32)
    demo_r2 = np.empty((N, 5), dtype=cl.cltypes.float)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)


    list_of_mass_all = np.array(list_of_mass,dtype=cl.cltypes.float)
    list_of_radius_and_velocity_all = np.array(list_of_radius_and_velocity, dtype=cl.cltypes.float)
    list_of_radius_and_velocity_new = np.zeros((N, 6), dtype=cl.cltypes.float)

    result = np.zeros((M, N, 6), dtype=cl.cltypes.float)
    a = np.zeros((N, 3), dtype=cl.cltypes.float)
    a_new = np.zeros((N, 3), dtype=cl.cltypes.float)

    mf = cl.mem_flags
    demo_buf = cl.Buffer(ctx, mf.WRITE_ONLY, demo_r.nbytes)
    demo_buf2 = cl.Buffer(ctx, mf.WRITE_ONLY, demo_r2.nbytes)
    buff_list = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=list_of_radius_and_velocity_all)
    buff_of_list_new = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=list_of_radius_and_velocity_new)
    buff_of_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    buff_of_a_new = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_new)
    buff_of_mass = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=list_of_mass_all)
    buff_of_result = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)
    T_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
    M_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M)
    N_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=N)
    tau_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau)


    prg = cl.Program(ctx,
                         """
                         float norm(__global float *list_of_radius_and_velocity, int i, int j)
                         {
                             double temp=0;
                             for (int k=0; k<3; ++k)
                                 temp+=(list_of_radius_and_velocity[6*i+k]-list_of_radius_and_velocity[6*j+k])*(list_of_radius_and_velocity[6*i+k]-list_of_radius_and_velocity[6*j+k]);
                             return sqrt(temp);
                         }
                         void acceleration(__global float *list_of_radius_and_velocity, __global float *list_of_mass_all,__global float *a, int N)
                         {
                             double MassSun = 1.99 * pow(10.0, 30);
                             double r_norm = 1.496 * pow(10.0, 11);
                             double G = 6.67 * pow(10.0,-11)*MassSun/pow(r_norm,3);
                            
                             for (int i=0; i<N; ++i)
                             {
                                 for (int k=0; k<3; ++k)
                                     a[3*i+k]=0;
                                 for (int j=0; j<N; ++j)
                                     if (i!=j)
                                         for (int k=0; k<3; ++k)
                                             a[3*i+k]+=G*list_of_mass_all[j]*(list_of_radius_and_velocity[6*j+k]-list_of_radius_and_velocity[6*i+k])/pow(norm(list_of_radius_and_velocity,i,j),3);
                             }
                         }
                         __kernel void verlet_cl(__global float *list_of_radius_and_velocity, __global float *list_of_mass_all, __global float *result, __global double *T_cl,__global double *tau_cl, __global int *M_cl, __global int *N_cl, __global float *a, __global float *a_new, __global float *list_of_radius_and_velocity_new)
                         {
                             double T=*T_cl;
                             
                             int M=*M_cl;
                             int N=*N_cl;            
                             
                             double tau=*tau_cl;
                             double MassSun = 1.99 * pow(10.0, 30);
                             double r_norm = 1.496 * pow(10.0, 11);
                             double G = 6.67 * pow(10.0,-11)*MassSun/pow(r_norm,3);
                            
                             for (int i=0; i<N; ++i)
                             {
                                 for (int k=0; k<3; ++k)
                                     a[3*i+k]=0;
                                 for (int j=0; j<N; ++j)
                                     if (i!=j)
                                         for (int k=0; k<3; ++k)
                                             a[3*i+k]+=G*list_of_mass_all[j]*(list_of_radius_and_velocity[6*j+k]-list_of_radius_and_velocity[6*i+k])/pow(norm(list_of_radius_and_velocity,i,j),3);
                             }
                             
  
                             for (int j=0; j<N; ++j)
                                 for (int k=0; k<6; ++k)
                                     result[6*j+k]=list_of_radius_and_velocity[6*j+k];
                                     
                             for (int i=1; i<M; ++i)
                             {
                                 for (int j=0; j<N; ++j)
                                     for (int k=0; k<3; ++k)
                                         {
                                            list_of_radius_and_velocity_new[6*j+k]=list_of_radius_and_velocity[6*j+k]+list_of_radius_and_velocity[6*j+k+3]*tau+0.5*a[3*j+k]*tau*tau;
                                         }
                                

                                         
                                 acceleration(list_of_radius_and_velocity_new,list_of_mass_all,a_new,N);
                                 for (int j=0; j<N; ++j)
                                     for (int k=0; k<3; ++k)
                                         list_of_radius_and_velocity_new[6*j+k+3]=list_of_radius_and_velocity[6*j+k+3]+0.5*(a[3*j+k]+a_new[3*j+k])*tau;
                                 for (int j=0; j<N; ++j)
                                     for (int k=0; k<3; ++k)
                                     {
                                         list_of_radius_and_velocity[6*j+k]=list_of_radius_and_velocity_new[6*j+k];
                                         list_of_radius_and_velocity_new[6*j+k]=0;
                                         list_of_radius_and_velocity[6*j+k+3]=list_of_radius_and_velocity_new[6*j+k+3];
                                         list_of_radius_and_velocity_new[6*j+k+3]=0;
                                         a[3*j+k]=a_new[3*j+k];
                                         result[6*N*i+6*j+k]=list_of_radius_and_velocity[6*j+k];
                                         result[6*N*i+6*j+k+3]=list_of_radius_and_velocity[6*j+k+3];
                                     }
                             }
                         }""")

    try:
        prg.build()
    except:
        print("Error:")
        print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise
    t = time.time()

    prg.verlet_cl(queue, (1,), None, buff_list, buff_of_mass, buff_of_result, T_cl,tau_cl, M_cl, N_cl, buff_of_a, buff_of_a_new, buff_of_list_new)
    cl.enqueue_read_buffer(queue, buff_of_result, result).wait()
    cl.enqueue_read_buffer(queue, buff_of_a, a).wait()


    return result

