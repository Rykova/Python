from __future__ import division
import numpy as np
import numpy.linalg as nlg
import copy
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
cimport cython
from cython.parallel import prange
@cython.boundscheck(False)
@cython.wraparound(False)


def Velocity_form_for_x(np.ndarray[DTYPE_t, ndim=2] list,np.ndarray[DTYPE_t, ndim=2] list_new, np.ndarray[DTYPE_t, ndim=1] a, double tau,int j):
    list_new[j,0:3]=list[j,0:3]+list[j,3:6]*tau+0.5*a*tau**2

def Velocity_form_for_v(np.ndarray[DTYPE_t, ndim=2] list,np.ndarray[DTYPE_t, ndim=2] list_new, np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] a_new, double tau,int j):
    list_new[j,3:6]=list[j,3:6]+0.5*(a+a_new)*tau

def accelaration_for_i_body(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] list_of_data, np.ndarray[DTYPE_t, ndim=1] list_of_mass, int i):
    cdef double G = 3.9644608161728576e-14
    cdef int N = list_of_data.shape[0]
    cdef int j
    for j in range(0,N):
        if i!=j:
            a[i]+=G*list_of_mass[j]*(list_of_data[j,0:3]-list_of_data[i,0:3])/nlg.norm(list_of_data[j,0:3]-list_of_data[i,0:3],2) ** 3


def cverletnotypedmemoryview(np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity, np.ndarray[DTYPE_t, ndim=1] list_of_mass, int M, double T):
    print "Verlet no typed memoryview"
    cdef int N=list_of_radius_and_velocity.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] result=np.zeros([M,N,6], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity_new = np.zeros([N,6],dtype=DTYPE)
    result[0] = copy.copy(list_of_radius_and_velocity)
    cdef double tau=T/M;
    cdef np.ndarray[DTYPE_t, ndim=2] a=np.zeros([N,3],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] a_new=np.zeros([N,3],dtype=DTYPE)
    cdef int i,j,k
    for i in range(0,N):
        accelaration_for_i_body(a, list_of_radius_and_velocity, list_of_mass, i)
    for i in range(1,M):
        list_of_radius_and_velocity_new = np.zeros((N,6))
        for j in range(0,N):
            Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)
        a_new = np.zeros((N, 3))
        for k in range(0, N):
            accelaration_for_i_body(a_new, list_of_radius_and_velocity_new, list_of_mass, k)
        for j in range(0,N):
            Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)
        list_of_radius_and_velocity=copy.copy(list_of_radius_and_velocity_new)
        a=copy.copy(a_new)
        result[i]=copy.copy(list_of_radius_and_velocity)
    return result

def norm(DTYPE_t[:,:] list_of_data, int i, int j):
    cdef int k
    cdef double temp=0
    for k in range(0,3):
        temp+=(list_of_data[i,k]-list_of_data[j,k])**2
    return temp ** 0.5

def accelaration_for_body_tm(DTYPE_t[:,:] a, DTYPE_t[:,:] list_of_data, DTYPE_t[:] list_of_mass):
    cdef double G = 3.9644608161728576e-14
    cdef int N = list_of_data.shape[0]
    cdef int j,k
    for i in range(0,N):
        for k in range(0,3):
            a[i,k]=0
        for j in range(0,N):
            if i!=j:
                for k in range(0,3):
                    a[i,k]+= G*list_of_mass[j]*(list_of_data[j,k]-list_of_data[i,k])/norm(list_of_data,i,j) ** 3

def Velocity_form_for_x_tm(DTYPE_t[:,:] list,DTYPE_t[:,:] list_new, DTYPE_t[:,:] a, double tau,int j):
    cdef int k
    for k in range(0,3):
        list_new[j,k]=list[j,k]+list[j,k+3]*tau+0.5*a[j,k]*tau**2

def Velocity_form_for_v_tm(DTYPE_t[:,:] list,DTYPE_t[:,:] list_new, DTYPE_t[:,:] a, DTYPE_t[:,:] a_new, double tau,int j):
    cdef int k
    for k in range(0,3):
        list_new[j,k+3]=list[j,k+3]+0.5*(a[j,k]+a_new[j,k])*tau

def cverlettypedmemoryview(np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity, np.ndarray[DTYPE_t, ndim=1] list_of_mass, int M, double T):
    print "Verlet typedmemoryview"
    cdef int N=list_of_radius_and_velocity.shape[0]
    cdef DTYPE_t[:,:,:] result=np.zeros([M,N,6], dtype=DTYPE)
    cdef DTYPE_t[:,:] list_of_radius_and_velocity_new = np.zeros([N,6],dtype=DTYPE)
    cdef double tau=T/M;
    cdef DTYPE_t[:,:] a=np.zeros([N,3],dtype=DTYPE)
    cdef DTYPE_t[:,:] a_new=np.zeros([N,3],dtype=DTYPE)
    cdef int i,j,k

    accelaration_for_body_tm(a, list_of_radius_and_velocity, list_of_mass)
    for i in range(1,M):
        for j in range(0,N):
            Velocity_form_for_x_tm(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a,tau,j)
        accelaration_for_body_tm(a_new, list_of_radius_and_velocity_new, list_of_mass)
        for j in range(0,N):
            Velocity_form_for_v_tm(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a,a_new,tau,j)
        for j in range(0,N):
            for k in range(0,3):
                list_of_radius_and_velocity[j,k]=list_of_radius_and_velocity_new[j,k]
                list_of_radius_and_velocity_new[j,k]=0
                list_of_radius_and_velocity[j,k+3]=list_of_radius_and_velocity_new[j,k+3]
                list_of_radius_and_velocity_new[j,k+3]=0
                a[j,k]=a_new[j,k]
                result[i,j,k]=list_of_radius_and_velocity[j,k]
                result[i,j,k+3]=list_of_radius_and_velocity[j,k+3]
    return result

def cverlet_openmp(np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity, np.ndarray[DTYPE_t, ndim=1] list_of_mass, int M, double T):
    print "Verlet no typed memoryview with openmp"
    cdef int N=list_of_radius_and_velocity.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] result=np.zeros([M,N,6], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity_new = np.zeros([N,6],dtype=DTYPE)
    result[0] = copy.copy(list_of_radius_and_velocity)
    cdef double tau=T/M;
    cdef np.ndarray[DTYPE_t, ndim=2] a=np.zeros([N,3],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] a_new=np.zeros([N,3],dtype=DTYPE)
    cdef int i,j,k
    #pragma parallel for
    for i in range(0,N):
        accelaration_for_i_body(a, list_of_radius_and_velocity, list_of_mass, i)
    for i in range(1,M):
        list_of_radius_and_velocity_new = np.zeros((N,6))
        #pragma parallel for
        for j in range(0,N):
            Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)
        a_new = np.zeros((N, 3))
        #pragma parallel for
        for k in range(0, N):
            accelaration_for_i_body(a_new, list_of_radius_and_velocity_new, list_of_mass, k)
       #pragma parallel for
        for j in range(0,N):
            Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)
        list_of_radius_and_velocity=copy.copy(list_of_radius_and_velocity_new)
        a=copy.copy(a_new)
        result[i]=copy.copy(list_of_radius_and_velocity)
    return result

def norm_openmp(DTYPE_t[:,:] list_of_data, int i, int j):
    cdef int k
    cdef double temp=0
    #pragma parallel for
    for k in range(0,3):
        temp+=(list_of_data[i,k]-list_of_data[j,k])**2
    return temp ** 0.5

def accelaration_for_body_tm_openmp(DTYPE_t[:,:] a, DTYPE_t[:,:] list_of_data, DTYPE_t[:] list_of_mass):
    cdef double G = 3.9644608161728576e-14
    cdef int N = list_of_data.shape[0]
    cdef int j,k
    #pragma parallel for
    for i in range(0,N):
        #pragma parallel for
        for k in range(0,3):
            a[i,k]=0
        for j in range(0,N):
            if i!=j:
                #pragma parallel for
                for k in range(0,3):
                    a[i,k]+= G*list_of_mass[j]*(list_of_data[j,k]-list_of_data[i,k])/norm_openmp(list_of_data,i,j) ** 3


def cverlettypedmemoryview_openmp(np.ndarray[DTYPE_t, ndim=2] list_of_radius_and_velocity, np.ndarray[DTYPE_t, ndim=1] list_of_mass, int M, double T):
    print "Verlet typed memoryview with openmp"
    cdef int N=list_of_radius_and_velocity.shape[0]
    cdef DTYPE_t[:,:,:] result=np.zeros([M,N,6], dtype=DTYPE)
    cdef DTYPE_t[:,:] list_of_radius_and_velocity_new = np.zeros([N,6],dtype=DTYPE)
    cdef double tau=T/M;
    cdef DTYPE_t[:,:] a=np.zeros([N,3],dtype=DTYPE)
    cdef DTYPE_t[:,:] a_new=np.zeros([N,3],dtype=DTYPE)
    cdef int i,j,k
    accelaration_for_body_tm_openmp(a, list_of_radius_and_velocity, list_of_mass)
    for i in range(1,M):
        #pragma parallel for
        for j in range(0,N):
            Velocity_form_for_x_tm(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a,tau,j)
        accelaration_for_body_tm_openmp(a_new, list_of_radius_and_velocity_new, list_of_mass)
        #pragma parallel for
        for j in range(0,N):
            Velocity_form_for_v_tm(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a,a_new,tau,j)
        #pragma parallel for
        for j in range(0,N):
            for k in range(0,3):
                list_of_radius_and_velocity[j,k]=list_of_radius_and_velocity_new[j,k]
                list_of_radius_and_velocity_new[j,k]=0
                list_of_radius_and_velocity[j,k+3]=list_of_radius_and_velocity_new[j,k+3]
                list_of_radius_and_velocity_new[j,k+3]=0
                a[j,k]=a_new[j,k]
                result[i,j,k]=list_of_radius_and_velocity[j,k]
                result[i,j,k+3]=list_of_radius_and_velocity[j,k+3]
    return result