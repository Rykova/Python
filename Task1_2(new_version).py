import sys
import numpy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt



class Triangles():
    def __init__(self, x1, x2, x3, y1, y2, y3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3


list_of_coord = []

x1 = [0, 0, 0]
x2 = [0, 5, 0]
x3 = [6, 5, 0]

y1 = [1, 4, 0]
y2 = [2, 4, 0]
y3 = [2, 3, 0]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [-1, 0, 0]
x2 = [0, -1, 0]
x3 = [0, 0, 0]

y1 = [0, 0, 0]
y2 = [0, 3, 0]
y3 = [5, 0, 0]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [-1, 0, 0]
x2 = [0, 2, 0]
x3 = [0, 0, 0]

y1 = [0, 0, 0]
y2 = [5, 0, 0]
y3 = [0, 4, 0]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 2, 0]
x3 = [1, 0, 0]

y1 = [0, -1, 0]
y2 = [0, 3, 0]
y3 = [7, -1, 0]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 4, 0]
x3 = [4, 0, 0]

y1 = [1, 2, 0]
y2 = [1, 1, -3]
y3 = [1/2, 2, -2]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 4, 0]
x3 = [4, 0, 0]

y1 = [4, 0, 0]
y2 = [1, 1, -3]
y3 = [1/2, 2, -2]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 4, 0]
x3 = [4, 0, 0]

y1 = [1, 2, 0]
y2 = [2, 1, 0]
y3 = [1/2, 2, -2]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 4, 0]
x3 = [4, 0, 0]

y1 = [-1, 2, 0]
y2 = [0, 2, 2]
y3 = [0, 0, -2]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

x1 = [0, 0, 0]
x2 = [0, 4, 0]
x3 = [5, 0, 0]

y1 = [1, 1, 2]
y2 = [5, 6, -2]
y3 = [3, -4, -1]

triang = Triangles(x1,x2,x3,y1,y2,y3)
list_of_coord.append(triang)

def IfIntersectionNormalMethod(x1,x2,x3,y1,y2,y3):
    #вычисляем уравнение плоскости для треугольника v0,v1,v2
    e1 = [y2[0] - y1[0], y2[1] - y1[1], y2[2] - y1[2]]
    e2 = [y3[0] - y1[0], y3[1] - y1[1], y3[2] - y1[2]]
    n1 = numpy.cross(e1, e2)
    d1 = -numpy.dot(n1, y1)
    eps = 1.401298E-45

     #положим x0,x1,x2 в уравнение плоскости, чтобы вычислить знак расстояния до плоскости
    du0 = numpy.dot(n1, x1) + d1
    du1 = numpy.dot(n1, x2) + d1
    du2 = numpy.dot(n1, x3) + d1

    #проверка компланарности
    if numpy.abs(du0) < eps:
        du0 = 0.0
    if numpy.abs(du1) < eps:
        du1 = 0.0
    if numpy.abs(du2) < eps:
        du2 = 0.0
    du0du1 = du0 * du1
    du0du2 = du0 * du2

   # все одинакового знака и не равны 0
    if du0du1 > 0.0 and du0du2 > 0.0:
        return 0

    #вычисляем плоскость для треугольника x0,x1,x2
    e1 = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]]
    e2 = [x3[0] - x1[0], x3[1] - x1[1], x3[2] - x1[2]]
    n2 = numpy.cross(e1, e2)
    d2 = -numpy.dot(n2, x1)

    dv0 = numpy.dot(n2, x1) + d2
    dv1 = numpy.dot(n2, x2) + d2
    dv2 = numpy.dot(n2, x3) + d2

    if numpy.abs(dv0) < eps:
        dv0 = 0.0
    if numpy.abs(dv1) < eps:
        dv1 = 0.0
    if numpy.abs(dv2) < eps:
        dv2 = 0.0

    dv0dv1 = dv0 * dv1
    dv0dv2 = dv0 * dv2

    if dv0dv1 > 0.0 and dv0dv2 > 0.0:
        return 0

    # вычислим направление пересечения
    dd = numpy.cross(n1, n2)
    #вычислим и проиндексируем самый большой компонент
    max = numpy.abs(dd[0])
    index = 0
    bb = numpy.abs(dd[1])
    cc =  numpy.abs(dd[2])
    if (bb > max):
        max = bb
        index = 1
    if (cc > max):
        max = cc
        index = 2

    vp0 = y1[index]
    vp1 = y2[index]
    vp2 = y3[index]

    up0 = x1[index]
    up1 = x2[index]
    up2 = x3[index]

    a,b,c,x0_new,x1_new = 0,0,0,0,0
    d,e,f,y0_new,y1_new =0,0,0,0,0
    if (ComputeIntervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0_new, x1_new)):
        return TriTriCoplanar(n1, y1, y2, y3, x1, x2, x3)
    if (ComputeIntervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0_new, y1_new)):
        return TriTriCoplanar(n1, y1, y2, y3, x1, x2, x3)


    xx = x0_new * x1_new
    yy = y0_new * y1_new
    xxyy = xx * yy


    tmp = a * xxyy
    isect1 = [0, 0]
    isect1[0] = tmp + b * x1_new * yy
    isect1[1] = tmp + c * x0_new * yy

    tmp = d * xxyy
    isect2 = [0, 0]
    isect2[0] = tmp + e * xx * y1_new
    isect2[1] = tmp + f * xx * y0_new

    Sort(isect1)
    Sort(isect2)

    return not(isect1[1] < isect2[0] or isect2[1] < isect1[0])


def ComputeIntervals(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A, B, C, X0, X1):
    if (D0D1 > 0.0): #D0,D1 на одной стороне от плоскости, D2 на другой
        A = VV2
        B = (VV0 - VV2) * D2
        C = (VV1 - VV2) * D2
        X0 = D2 - D0
        X1 = D2 - D1
    else:
        if (D0D2 > 0.0):
            A = VV1
            B = (VV0 - VV1) * D1
            C = (VV2 - VV1) * D1
            X0 = D1 - D0
            X1 = D1 - D2
        else:
            if (D1 * D2 > 0.0 or D0 != 0.0):
                A = VV0
                B = (VV1 - VV0) * D0
                C = (VV2 - VV0) * D0
                X0 = D0 - D1
                X1 = D0 - D2
            else:
                if (D1 != 0.0):
                    A = VV1
                    B = (VV0 - VV1) * D1
                    C = (VV2 - VV1) * D1
                    X0 = D1 - D0
                    X1 = D1 - D2
                else:
                    if (D2 != 0.0):
                        A = VV2
                        B = (VV0 - VV2) * D2
                        C = (VV1 - VV2) * D2
                        X0 = D2 - D0
                        X1 = D2 - D1
                    else:
                        return 1

    return 0


#проекция плоскости
def TriTriCoplanar(N,v0, v1, v2, u0, u1, u2):
    A = [0,0,0]
    A[0] = numpy.abs(N[0]);
    A[1] = numpy.abs(N[1]);
    A[2] = numpy.abs(N[2]);
    if (A[0] > A[1]):
        if (A[0] > A[2]):
            i0 = 1
            i1 = 2
        else:
            i0 = 0
            i1 = 1
    else:
        if (A[2] > A[1]):
            i0 = 0
            i1 = 1
        else:
            i0 = 0
            i1 = 2

    if (EdgeAgainstTriEdges(v0, v1, u0, u1, u2, i0, i1)):
        return 1

    if (EdgeAgainstTriEdges(v1, v2, u0, u1, u2, i0, i1)):
        return  1
    if (EdgeAgainstTriEdges(v2, v0, u0, u1, u2, i0, i1)):
        return  1
    if (PointInTri(v0, u0, u1, u2, i0, i1)):
        return  1
    if (PointInTri(u0, v0, v1, v2, i0, i1)):
        return  1

    return 0


def EdgeAgainstTriEdges(v0,v1,u0,u1,u2,i0,i1):
    if (EdgeEdgeTest(v0, v1, u0, u1, i0, i1)): #пересекаются u0,u1 с v0,v1
        return  1
    if (EdgeEdgeTest(v0, v1, u1, u2, i0, i1)): # u1,u1 c v0,v1
        return  1
    if (EdgeEdgeTest(v0, v1, u2, u0, i0, i1)): # u2,u0 c v0,v1
        return  1
    return  0


#Быстрое пересечение сегментов линии
def EdgeEdgeTest(v0,v1,u0,u1,i0,i1):
    Ax = v1[i0] - v0[i0]
    Ay = v1[i1] - v0[i1]

    Bx = u0[i0] - u1[i0]
    By = u0[i1] - u1[i1]
    Cx = v0[i0] - u0[i0]
    Cy = v0[i1] - u0[i1]
    f = Ay * Bx - Ax * By
    d = By * Cx - Bx * Cy
    if ((f > 0 and d >= 0 and d <= f) or (f < 0 and d <= 0 and d >= f)):
        e = Ax * Cy - Ay * Cx;
        if (f > 0):
            if (e >= 0 and e <= f):
                return 1
        else:
            if (e <= 0 and e >= f):
                return  1

    return 0


# точка v0 внутри  треугольника (u0,u1,u2)?
# i0,i1 индексы
def PointInTri(v0,u0,u1,u2,i0,i1):
    a = u1[i1] - u0[i1]
    b = -(u1[i0] - u0[i0])
    c = -a * u0[i0] - b * u0[i1]
    d0 = a * v0[i0] + b * v0[i1] + c

    a = u2[i1] - u1[i1]
    b = -(u2[i0] - u1[i0])
    c = -a * u1[i0] - b * u1[i1]
    d1 = a * v0[i0] + b * v0[i1] + c

    a = u0[i1] - u2[i1]
    b = -(u0[i0] - u2[i0])
    c = -a * u2[i0] - b * u2[i1]
    d2 = a * v0[i0] + b * v0[i1] + c

    if (d0 * d1 > 0.0):
        if (d0 * d2 > 0.0):
            return  1

    return  0

def Sort(v):
    if (v[0] > v[1]):
        c = v[0];
        v[0] = v[1];
        v[1] = c;




for elem in list_of_coord:
    if (IfIntersectionNormalMethod(elem.x1,elem.x2,elem.x3,elem.y1,elem.y2,elem.y3)):
        print("Пересекаются")
    else:
        print("Не пересекаются")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    numpy.set_printoptions(threshold='nan')

    a = numpy.zeros((2, 9))
    for i in range(3):
        a[0,i] = elem.x1[i]
        a[0,i+3] = elem.x2[i]
        a[0,i+6] = elem.x3[i]
        a[1,i] = elem.y1[i]
        a[1,i+3] = elem.y2[i]
        a[1,i+6] = elem.y3[i]

    fc = ["crimson" if i%2 else "gold" for i in range(a.shape[0])]

    poly3d = [[ a[i, j*3:j*3+3] for j in range(3)  ] for i in range(a.shape[0])]

    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, linewidths=1))

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    plt.show()
