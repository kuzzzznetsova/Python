import numpy as np

print('   Лабораторная работа №1')
print('     Метод вращений')
print('Кузнецова Дарина М8О-305Б-20')


def max_element_of_matrix(A):
    max1 = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j and max1 < abs(A[i][j]):
                max1 = abs(A[i][j])
                max_i = i
                max_j = j
    return max_i, max_j


def transpose(A):
    AT = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            AT[j][i] = A[i][j]
    return AT


def multiply(A, B):
    C = np.zeros((len(A), len(B)))
    for i in range(len(C)):
        for j in range(len(C)):
            for k in range(len(C)):
                C[i][j] += A[i][k] * B[k][j]
    return C

def check_simmetric(A):
    k = len(A)
    k = k * k
    k1 = 0
    A_tr = transpose(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] == A_tr[i][j]:
                k1 = k1 + 1
    if k == k1:
        print("Матрица А - является симметрической")
    else:
        print("Матрица А - не является симметрической")


def rotation_method(A):
    m = 3
    e = 0.01
    T = []
    A1 = np.copy(A)
    while (m > e):
        m = 0
        A0 = np.copy(A1)
        iMax, jMax = max_element_of_matrix(A0)

        phi = np.arctan(2 * A0[iMax][jMax] / (A0[iMax][iMax] - A0[jMax][jMax])) / 2
        T0 = np.zeros((len(A), len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                if i == iMax and j == jMax:
                    T0[i][j] = -np.sin(phi)
                    T0[i][i] = np.cos(phi)
                    T0[j][j] = np.cos(phi)
                    T0[j][i] = np.sin(phi)
                elif i == j and i != iMax and j != jMax:
                    T0[i][j] = 1
        T.append(T0)
        T0T = transpose(T0)
        A1 = multiply(multiply(T0T, A0), T0)
        # критерий окончания (корень из суммы квадратов)
        for i in range(len(A)):
            for j in range(len(A)):
                if i < j:
                    m += A1[i][j] * A1[i][j]
        m = np.sqrt(m)
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                print("Собственное значение lyambda(", i, ") = ", np.around(A1[i][j], 2))
    for i in range(len(T) - 1):
        T[i + 1] = multiply(T[i], T[i + 1])

    T = T[-1]
    for i in range(len(A)):
        print("Собственный вектор x(", i + 1, ")")
        for j in range(len(A)):
            print(np.around(T[j][i], 2), sep='\n')

    return np.around(A1, 2)


A = [[-6, 6, -8], [6, -4, 9], [-8, 9, -2]]
check_simmetric(A)
print("Матрица А:\n", np.around(rotation_method(A), 2))
#проверка на симметричность матрицы
# -6 6 -8
# 6 -4  9
# -8 9 -2
