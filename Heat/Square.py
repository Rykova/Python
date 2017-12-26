import Task1_2
import numpy as np
import Parser
import scipy.optimize as so
import scipy.integrate as si
import copy

def huge_intersect(P1, P2, P3, P4, P5, P6):
    if Task1_2.IfIntersectionNormalMethod(P1,P2,P3,P4,P5,P6)==1:
        if abs(Task1_2.deter(P1,P2,P3,P4))>10**(-6):
            return False
        if abs(Task1_2.deter(P1,P2,P3,P5))>10**(-6):
            return False
        if abs(Task1_2.deter(P1,P2,P3,P6))>10**(-6):
            return False
        return True
    return False
def little_S(hx,hy):
    l1=np.linalg.norm(hx)
    l2=np.linalg.norm(hy)
    l3=np.linalg.norm(hx-hy)
    p=(l1+l2+l3)/2
    return 2*(p*(p-l1)*(p-l2)*(p-l3))**0.5
def triangle_IS(P1, P2, P3, P4, P5, P6):
    if not huge_intersect(P1,P2,P3,P4,P5,P6):
        return 0.0
    N=100
    S=0.0
    x=P3-P1
    y=P2-P1
    hx=x/N
    hy=y/N
    Sh=little_S(hx,hy)
    for i in range(N):
        for j in range(N-i):
            temp=0
            pp=[P1+i*hx+j*hy, P1+(i+1)*hx+j*hy, P1+i*hx+(j+1)*hy, P1+(i+1)*hx+(j+1)*hy]
            for P in pp:
                if Task1_2.internal(P4,P5,P6,P)==1:
                    temp+=0.25
            if j<N-i-1:
                S+=temp*Sh
            else:
                S+=temp*Sh/2
    return S
def triangle_S(lst,mind):
    l1=np.linalg.norm(lst[mind[0]]-lst[mind[1]])
    l2=np.linalg.norm(lst[mind[0]]-lst[mind[2]])
    l3=np.linalg.norm(lst[mind[2]]-lst[mind[1]])
    p=(l1+l2+l3)/2
    return (p*(p-l1)*(p-l2)*(p-l3))**0.5
def intersection_S(glob,locind,i,j):
    S=0
    for ind1 in locind[i]:
        for ind2 in locind[j]:
            P1=glob[i][ind1[0]]
            P2=glob[i][ind1[1]]
            P3=glob[i][ind1[2]]
            P4=glob[j][ind2[0]]
            P5=glob[j][ind2[1]]
            P6=glob[j][ind2[2]]
            S+=triangle_IS(P1,P2,P3,P4,P5,P6)
    return S
def object_S(glob,locind,i):
    S=0.0
    for el in locind[i]:
        S+=triangle_S(glob[i],el)
    return S
def create_file_with_square(vel,face):
    square=np.zeros((len(vel),len(vel)))
    f=open('square.txt','w')
    for i in range(len(vel)):
        for j in range(len(vel)):
            print(i,j)
            if i==j:
                square[i,j]=object_S(vel,face,i)
            else:
                square[i,j]=intersection_S(vel,face,i,j)
            f.write(str(square[i][j]))
            f.write(' ')
        f.write('\n')
    f.close()
    return square
