import sys
import numpy

x1 = [10, 0, 0]
x2 = [0, 10, 0]
x3 = [0, 0, 0]

y1 = [1, 1, 0]
y2 = [2, 2, 0]
y3 = [1, 2, 0]

def IfIntersectionNormalMethod():
    e1 = [y2[0] - y1[0], y2[1] - y1[1], y2[2] - y1[2]]
    e2 = [y3[0] - y1[0], y3[1] - y1[1], y3[2] - y1[2]]
    n1 = numpy.cross(e1, e2)
    d1 = -numpy.dot(n1, y1)
    eps = 1.401298E-45

    du0 = numpy.dot(n1, x1) + d1
    du1 = numpy.dot(n1, x2) + d1
    du2 = numpy.dot(n1, x3) + d1
    if numpy.abs(du0) < eps:
        du0 = 0.0
    if numpy.abs(du1) < eps:
        du1 = 0.0
    if numpy.abs(du2) < eps:
        du2 = 0.0
    du0du1 = du0 * du1
    du0du2 = du0 * du2


    if du0du1 > 0.0 and du0du2 > 0.0:
        return 0

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

    dd = numpy.cross(n1, n2)
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
    if (D0D1 > 0.0):
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
    if (EdgeEdgeTest(v0, v1, u0, u1, i0, i1)):
        return  1
    if (EdgeEdgeTest(v0, v1, u1, u2, i0, i1)):
        return  1
    if (EdgeEdgeTest(v0, v1, u2, u0, i0, i1)):
        return  1
    return  0


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


if(IfIntersectionNormalMethod()):
    print("Пересекаются")
else:
    print("Не пересекаются")