import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import math
import copy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from mainApp.models import *
from mainApp.views import *
from django.core.files.base import ContentFile

mas = search()
K = Matrixsize.objects.all().order_by('-id_km')[0].id_k
M = Matrixsize.objects.all().order_by('-id_km')[0].id_m
R = mas[2]
hr = R / (M + (0.5))
hfi = math.pi / K
nu = mas[0]
#nu = Nu.objects.all().order_by('-id_nu')[0].value_nu
q = mas[1]
D0 = 0
D1 = 100

be = backup()


def main():
    X = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    Y = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixU_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixD_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixU_new, matrixD_new = FindMatrixUandD(0.01)
    for i in range(M + 3):
        for j in range(K + 3):
            X[i][j] = (i + 0.5) * hr * math.cos(j * hfi)
            Y[i][j] = (i + 0.5) * hr * math.sin(j * hfi)
    # xgrid, ygrid = numpy.meshgrid(X, Y)
    xgrid = numpy.array(X)
    ygrid = numpy.array(Y)
    zgrid = numpy.array(matrixU_new)
    xl = len(X)
    yl = len(Y)
    zl = len(matrixU_new)
    return xgrid, ygrid, zgrid


def FindMatrixUandD(alpha):
    a = (math.pow(hr, 2) * math.pow(hfi, 2)) / (math.pow(hr, 2) + math.pow(hfi, 2))
    b = a
    fi = 0
    psi = 0
    matrixU_old = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixV_old = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixD_old = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixU_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixV_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixD_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    # sqrt = 0
    # newsqrt = 0;
    eps = 0.000001
    # alpha = 0.2
    count = 0
    # normU = 0
    # normV = 0
    # normD = 0
    # newnormU = 1
    # newnormV = 1
    # newnormD = 1

    # for alpha in numpy.arange(0.001, 1, 0.001):
    normD = 0
    newnormD = 1
    NachalnayMatrix1(matrixD_old)
    NachalnayMatrix1(matrixD_new)
    while (abs(newnormD - normD) >= eps):
        matrixD_old = copy.deepcopy(matrixD_new)
        normD = newnormD
        normU = 0
        normV = 0
        newnormU = 1
        newnormV = 1
        NachalnayMatrix0(matrixU_old)
        NachalnayMatrix0(matrixV_old)
        NachalnayMatrix0(matrixU_new)
        NachalnayMatrix0(matrixV_new)
        while (abs(newnormU - normU) >= eps and abs(newnormV - normV) >= eps):
            matrixU_old = copy.deepcopy(matrixU_new)
            matrixV_old = copy.deepcopy(matrixV_new)

            normU = newnormU
            normV = newnormV

            PerekidPoM(matrixU_old, matrixV_old, matrixD_old)
            PerekidPoK(matrixU_old, matrixV_old, matrixD_old)

            matrixU_new = progU(matrixU_old, matrixV_old, matrixD_old)
            matrixV_new = progV(matrixV_old, matrixU_new, matrixD_old)

            # // Console.WriteLine("новая U");

            # // print(matrixU_new);

            # // Console.WriteLine("итерация №{0}", count);
            newnormU = norma(matrixU_new, matrixU_old)
            newnormV = norma(matrixV_new, matrixV_old)
            count += 1
            # print("нашлась U", count)
        matrixD_new = NahogdD(matrixD_old, matrixU_new, alpha)

        # // Console.WriteLine("новая D");
        # // print(matrixD_new);

        newnormD = norma(matrixD_new, matrixD_old)
        count += 1
        print("нашлась D", count)

    print("новая U")
    print(matrixU_new)
    print("новая D")
    print(matrixD_new)
    print("------------------------------------")
    print(alpha)
    print("итераций №{0}", count)
    return matrixU_new, matrixD_new


def makeData():
    x = numpy.arange(0, M + 3, 1)
    y = numpy.arange(0, K + 3, 1)
    matrixU_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixD_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    FindMatrixUandD(0)
    matrixU_new, matrixD_new = FindMatrixUandD(0)
    xgrid, ygrid = numpy.meshgrid(y, x)
    zgrid = numpy.array(matrixD_new)
    xl = len(xgrid)
    yl = len(ygrid)

    # zgrid = numpy.sin(xgrid) * numpy.sin(ygrid) / (xgrid * ygrid)
    zl = len(matrixU_new)
    return xgrid, ygrid, zgrid


def func_rr(F_k_mp1, F_k_m, F_k_mm1):
    return (F_k_mp1 - 2 * F_k_m + F_k_mm1) / (hr * hr)


def func_fifi(F_kp1_m, F_k_m, F_km1_m):
    return (F_kp1_m - 2 * F_k_m + F_km1_m) / (hfi * hfi)


def func_rfi(F_kp1_mp1, F_kp1_mm1, F_km1_mp1, F_km1_mm1):
    return (F_kp1_mp1 - F_kp1_mm1 - F_km1_mp1 + F_km1_mm1) / (4 * hr * hfi)


def func_r(F_k_mp1, F_k_mm1):
    return (F_k_mp1 - F_k_mm1) / (2 * hr)


def func_rM(F_k_m, F_k_mm1, F_k_mm2):
    return (3 * F_k_m - 4 * F_k_mm1 + F_k_mm2) / (2 * hr)


def func_fi(F_kp1_m, F_km1_m):
    return (F_kp1_m - F_km1_m) / (2 * hfi)


def laplas(matrix, i, j):
    rm = (i + 0.5) * hr
    return func_rr(matrix[i + 1][j], matrix[i][j], matrix[i - 1][j]) + \
           func_fifi(matrix[i][j + 1], matrix[i][j], matrix[i][j - 1]) / math.pow(rm, 2) + \
           func_r(matrix[i + 1][j], matrix[i - 1][j]) / rm


def laplasian(matrix, i, j, F):
    return hfi * hfi * (2 * i / (2 * i + 1)) * matrix[i - 1][j] - 2 * \
           (hfi * hfi + (4 / (2 * i + 1) / (2 * i + 1))) * matrix[i][j] + \
           hfi * hfi * ((2 * i + 2) / (2 * i + 1)) * matrix[i + 1][j] + \
           (4 / (2 * i + 1) / (2 * i + 1)) * (matrix[i][j + 1] + matrix[i][j - 1]) - \
           (hr * hr * hfi * hfi * F)


def yravn(matrixD1, matrixU1, i, j):
    rm = (i + 0.5) * hr
    return (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[i - 1][j]) * \
            func_r(matrixU1[i + 1][j], matrixU1[i - 1][j]) + \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[i - 1][j]) * \
            func_r(matrixD1[i + 1][j], matrixD1[i - 1][j])) / rm + \
           (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[i - 1][j]) * \
            func_fifi(matrixU1[i][j + 1], matrixU1[i][j], matrixU1[i][j - 1]) - \
            2 * func_rfi(matrixD1[i + 1][j + 1], matrixD1[i - 1][j + 1], matrixD1[i + 1][j - 1],
                         matrixD1[i - 1][j - 1]) * \
            func_rfi(matrixU1[i + 1][j + 1], matrixU1[i - 1][j + 1], matrixU1[i + 1][j - 1], matrixU1[i - 1][j - 1]) + \
            func_fifi(matrixD1[i][j + 1], matrixD1[i][j], matrixD1[i][j - 1]) * \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[i - 1][j])) / math.pow(rm, 2) + \
           2 * (func_rfi(matrixD1[i + 1][j + 1], matrixD1[i - 1][j + 1], matrixD1[i + 1][j - 1],
                         matrixD1[i - 1][j - 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1]) +
                func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_rfi(matrixU1[i + 1][j + 1], matrixU1[i - 1][j + 1], matrixU1[i + 1][j - 1],
                         matrixU1[i - 1][j - 1])) / math.pow(rm, 3) - \
           2 * (func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1])) / math.pow(rm, 4)


def yravn0(matrixD1, matrixU1, i, j):
    rm = (i + 0.5) * hr;
    return (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[0][K + 2 - j]) * \
            func_r(matrixU1[i + 1][j], matrixU1[0][K + 2 - j]) + \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[0][K + 2 - j]) * \
            func_r(matrixD1[i + 1][j], matrixD1[0][K + 2 - j])) / rm + \
           (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[0][K + 2 - j]) * \
            func_fifi(matrixU1[i][j + 1], matrixU1[i][j], matrixU1[i][j - 1]) - \
            2 * func_rfi(matrixD1[i + 1][j + 1], matrixD1[0][K + 2 - j - 1], matrixD1[i + 1][j - 1],
                         matrixD1[0][K + 2 - j + 1]) * \
            func_rfi(matrixU1[i + 1][j + 1], matrixU1[0][K + 2 - j - 1], matrixU1[i + 1][j - 1],
                     matrixU1[0][K + 2 - j + 1]) + \
            func_fifi(matrixD1[i][j + 1], matrixD1[i][j], matrixD1[i][j - 1]) * \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[0][K + 2 - j])) / math.pow(rm, 2) + \
           2 * (func_rfi(matrixD1[i + 1][j + 1], matrixD1[0][K + 2 - j - 1], matrixD1[i + 1][j - 1],
                         matrixD1[0][K + 2 - j + 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1]) + \
                func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_rfi(matrixU1[i + 1][j + 1], matrixU1[0][K + 2 - j - 1], matrixU1[i + 1][j - 1],
                         matrixU1[0][K + 2 - j - 1])) / math.pow(rm, 3) - \
           2 * (func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1])) / math.pow(rm, 4)


def yravnM(matrixD1, matrixU1, i, j):
    rm = (i + 0.5) * hr
    return (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[0][K + 2 - j]) * \
            func_r(matrixU1[i + 1][j], matrixU1[0][K + 2 - j]) + \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[0][K + 2 - j]) * \
            func_r(matrixD1[i + 1][j], matrixD1[0][K + 2 - j])) / rm + \
           (func_rr(matrixD1[i + 1][j], matrixD1[i][j], matrixD1[0][K + 2 - j]) * \
            func_fifi(matrixU1[i][j + 1], matrixU1[i][j], matrixU1[i][j - 1]) - \
            2 * func_rfi(matrixD1[i + 1][j + 1], matrixD1[0][K + 2 - j - 1], matrixD1[i + 1][j - 1],
                         matrixD1[0][K + 2 - j + 1]) * \
            func_rfi(matrixU1[i + 1][j + 1], matrixU1[0][K + 2 - j - 1], matrixU1[i + 1][j - 1],
                     matrixU1[0][K + 2 - j + 1]) + \
            func_fifi(matrixD1[i][j + 1], matrixD1[i][j], matrixD1[i][j - 1]) * \
            func_rr(matrixU1[i + 1][j], matrixU1[i][j], matrixU1[0][K + 2 - j])) / math.pow(rm, 2) + \
           2 * (func_rfi(matrixD1[i + 1][j + 1], matrixD1[0][K + 2 - j - 1], matrixD1[i + 1][j - 1],
                         matrixD1[0][K + 2 - j + 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1]) + \
                func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_rfi(matrixU1[i + 1][j + 1], matrixU1[0][K + 2 - j - 1], matrixU1[i + 1][j - 1],
                         matrixU1[0][K + 2 - j - 1])) / math.pow(rm, 3) - \
           2 * (func_fi(matrixD1[i][j + 1], matrixD1[i][j - 1]) * \
                func_fi(matrixU1[i][j + 1], matrixU1[i][j - 1])) / math.pow(rm, 4)


def NachalnayMatrix0(matrix):
    # matrix = [[0 for y in range(0,M + 3,1)] for x in range(0,K + 3,1)]
    for i in range(M + 3):
        for j in range(K + 3):
            matrix[i][j] = 0


def NachalnayMatrix1(matrix):
    # matrix = [[1 for y in range(0,M + 3,1)] for x in range(0,K + 3,1)]
    for i in range(M + 3):
        for j in range(K + 3):
            matrix[i][j] = 1


def PerekidPoK(matrixU, matrixV, matrixD):
    for i in range(M + 3):
        for j in range(K + 3):
            if (j == 2):
                matrixU[i][0] = matrixU[i][j]
                matrixV[i][0] = matrixV[i][j]
                matrixD[i][0] = matrixD[i][j]
            elif (j == K):
                matrixU[i][K + 2] = matrixU[i][j]
                matrixV[i][K + 2] = matrixV[i][j]
                matrixD[i][K + 2] = matrixD[i][j]


def PerekidPoM(matrixU, matrixV, matrixD):
    for i in range(0, 3, 1):
        for j in range(0, (K + 3), 1):
            if i == 2 and j >= 1:
                k = K + 2 - j
                matrixU[0][k] = matrixU[i][j]
                matrixV[0][k] = matrixV[i][j]
                matrixD[0][k] = matrixD[i][j]


def norma(matrix_new, matrix_old):
    maxi = 0
    max = 0
    for i in range(M + 3):
        for j in range(K + 3):
            maxi = abs(matrix_new[i][j] - matrix_old[i][j])
            if (maxi >= max):
                max = maxi
    return max


def progU(matrixU, matrixV, matrixD):
    matrixA = [[0 for j in range(0, M + 3, 1)] for i in range(0, M + 3, 1)]
    matrixW = [0 for j in range(0, M + 3, 1)]
    matrixF = [0 for j in range(0, M + 3, 1)]
    matrix = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrix = copy.deepcopy(matrixU)
    F = 0
    for j in range(1, K + 2):
        for i in range(M + 3):
            D = matrixD[i][j]
            F = matrixV[i][j] / D
            if (i == 0):
                matrixA[i][i] = -2.0 * (hfi * hfi + (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)))
                matrixA[i][i + 1] = hfi * hfi * ((2.0 * i + 2.0) / (2.0 * i + 1.0))
                matrixW[i] = matrix[i][j]
                matrixF[i] = -1.0 * (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)) * (matrix[i][j + 1] + matrix[i][j - 1]) + \
                             (hr * hr * hfi * hfi * F)

            elif (i == M + 2):
                matrixA[i][i - 1] = 0.0
                matrixA[i][i] = 1.0
                matrixW[i] = matrix[i][j]
                matrixF[i] = 0.0

            else:
                matrixA[i][i - 1] = hfi * hfi * (2.0 * i / (2.0 * i + 1.0))
                matrixA[i][i] = -2.0 * (hfi * hfi + (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)))
                matrixA[i][i + 1] = hfi * hfi * ((2.0 * i + 2.0) / (2.0 * i + 1.0))
                matrixW[i] = matrix[i][j]
                matrixF[i] = -1.0 * (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)) * (matrix[i][j + 1] + matrix[i][j - 1]) + \
                             (hr * hr * hfi * hfi * F)

        progonka(matrixA, matrixF, matrixW, M + 3)
        for i in range(M + 3):
            matrix[i][j] = matrixW[i]

    return matrix


def progV(matrixV, matrixU, matrixD):
    matrixA = [[0 for j in range(0, M + 3, 1)] for i in range(0, M + 3, 1)]
    matrixW = [0 for j in range(0, M + 3, 1)]
    matrixF = [0 for j in range(0, M + 3, 1)]
    matrix = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrix = copy.deepcopy(matrixV)
    F = 0
    for j in range(1, K + 2):
        for i in range(M + 3):
            if (i == 0):
                F = (1.0 - nu) * yravn0(matrixD, matrixU, i, j) + q
                matrixA[i][i] = -2.0 * (hfi * hfi + (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)))
                matrixA[i][i + 1] = hfi * hfi * ((2.0 * i + 2.0) / (2.0 * i + 1.0))
                matrixW[i] = matrix[i][j]
                matrixF[i] = -1.0 * (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)) * (matrix[i][j + 1] + matrix[i][j - 1]) + \
                             (hr * hr * hfi * hfi * F)
            elif (i == M + 2):
                matrixA[i][i - 1] = hfi * hfi * (2.0 * i / (2.0 * i + 1.0))
                matrixA[i][i] = -2.0 * (hfi * hfi + (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)))
                matrixW[i] = matrix[i][j]
                matrixF[i] = (1.0 - nu) * func_rM(matrixU[i][j], matrixU[i - 1][j], matrixU[i - 2][j]) / R
            else:
                F = (1.0 - nu) * yravn(matrixD, matrixU, i, j) + q
                matrixA[i][i - 1] = hfi * hfi * (2.0 * i / (2.0 * i + 1.0))
                matrixA[i][i] = -2.0 * (hfi * hfi + (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)))
                matrixA[i][i + 1] = hfi * hfi * ((2.0 * i + 2.0) / (2.0 * i + 1.0))
                matrixW[i] = matrix[i][j]
                matrixF[i] = -1.0 * (4.0 / (2.0 * i + 1.0) / (2.0 * i + 1.0)) * (matrix[i][j + 1] + matrix[i][j - 1]) + \
                             (hr * hr * hfi * hfi * F)

        progonka(matrixA, matrixF, matrixW, M + 3)
        for i in range(M + 3):
            matrix[i][j] = matrixW[i]

    return matrix


def progonka(A, B, X, n):
    m = n + 1
    C = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j]
        C[i][n] = B[i]

    for k in range(n - 1):
        for i in range(k + 1, n):
            for j in range(m - 1, k + 1, -1):
                C[i][j] = C[i][j] - C[i][k] * C[k][j] / C[k][k]

    X[n - 1] = C[n - 1][m - 1] / C[n - 1][m - 2]

    for i in range(n - 2, -1, -1):
        s = 0
        for j in range(i + 1, m - 1):
            s = s + C[i][j] * X[j]
        X[i] = (C[i][m - 1] - s) / C[i][i]


# def copy(matrix):
#   newmatrix = []
#  for i in range(M + 3):
#     for j in range(K + 3):
#        newmatrix[i, j] = matrix[i, j]
# return newmatrix


def funcZ(i, j):
    return 1 / 8 * math.pow(1 - (i + 0.5) * hr, 2)


def NahogdD(OldMatrixD, matrixU, alpha):
    NewMatrixD = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    OldMatrixD[0][0] = OldMatrixD[0][1] + OldMatrixD[1][0] - OldMatrixD[1][1]
    f = 0
    for i in range(M + 3):
        for j in range(K + 3):
            f = OldMatrixD[i][j] + alpha * (matrixU[i][j] - funcZ(i, j))
            if f < D0:
                NewMatrixD[i][j] = D0
            elif D0 <= f <= D1:
                NewMatrixD[i][j] = f
            elif f > D1:
                NewMatrixD[i][j] = D1
    return NewMatrixD


def plot3d():
    fig = plt.figure()
    matrixU_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    matrixD_new = [[0 for j in range(0, K + 3, 1)] for i in range(0, M + 3, 1)]
    FindMatrixUandD(0)
    matrixU_new, matrixD_new = FindMatrixUandD(0)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(M + 3)
    Y = np.arange(K + 3)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.array(matrixU_new)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)

    plt.show()


# main()
# print("end")
def printImage():
    x, y, z = main()
    fig = pylab.figure(figsize=(8, 8))
    axes = Axes3D(fig)
    axes.elev, axes.azim = 20, -40
    axes.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
    # axes.plot(x,y)
    # pylab.show()

    pylab.savefig('mainApp/media/profile_image/image' + str(
        (Graph.objects.all().order_by('-id_graph')[0].id_graph) + 1) + '.png')


    p = Graph(images_graph='profile_image/image' + str(
        (Graph.objects.all().order_by('-id_graph')[0].id_graph) + 1) + '.png')

    p.save()
    return p.id_graph
    # plot3d()
