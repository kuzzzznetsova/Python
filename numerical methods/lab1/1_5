import numpy as np


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0


def transpose(A): #транспонирование
    AT = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            AT[j][i] = A[i][j]
    return AT


def multiply(A, B): #умножение
    C = np.zeros((len(A), len(B)))
    for i in range(len(C)):
        for j in range(len(C)):
            for k in range(len(C)):
                C[i][j] += A[i][k] * B[k][j]
    return C


def division(A, B): #деление
    C = np.zeros((len(A), len(A)))
    for i in range(len(C)):
        for j in range(len(C)):
            C[i, j] = A[i, j] / B
    return C


def difference(A, B): #разность
    C = np.zeros((len(A), len(A)))
    for i in range(len(C)):
        for j in range(len(C)):
            C[i, j] = A[i, j] - B[i, j]
    return C


def norm_vector(b): #норма (корень из суммы квадратов
    norm = 0
    for i in range(len(b)):
        norm += b[i] * b[i]
    return np.sqrt(norm)


def Householder_matrix(H, index):
    H1 = np.copy(H)
    x1 = np.zeros(len(H))
    b = np.zeros(len(H))
    for k in range(len(H)):
        b[k] = H1[k, index]
    norm = norm_vector(b)
    for i in range(len(H)):
        if i == index:
            x1[i] = H1[i, index] + sign(H1[i, index]) * norm
        elif i < index:
            x1[i] = 0
        else:
            x1[i] = H1[i, index]
    v1 = np.zeros((len(x1), len(x1)))
    for i in range(len(v1)):
        v1[i, 0] = x1[i]
    v1_T = transpose(v1)
    E = np.eye(len(H))
    H1 = difference(E, 2 * (division(multiply(v1, v1_T), multiply(v1_T, v1)[0, 0])))
    return H1


def QR(A):
    A0 = np.copy(A)
    H = []
    for i in range(len(A) - 1):
        H0 = Householder_matrix(A0, i)
        A0 = multiply(H0, A0)
        H.append(H0)
    for i in range(len(H) - 1):
        H[i + 1] = multiply(H[i], H[i + 1])
    return A0, H[-1]


def eigenvalues(A): #собственный вектор
    e = 0.01
    m = 1
    A0 = np.copy(A)
    while (m > e):
        R0, Q0 = QR(A0)
        a = []
        A0 = multiply(R0, Q0)
        for i in range(len(A)):
            for j in range(len(A)):
                if i > j:
                    a.append(A0[i, j])
        m = norm_vector(a)
    L = []
    for i in range(len(A)):
        L.append(A0[i, i])
    return L


A = np.array([[9, 0, 2],
              [-6, 4, 4],
              [-2, -7, 5]])
R, Q = QR(A)
print("Матрица Q:\n", Q)
print("Матрица R:\n", R)
print("Собственные значения матрицы А:\n", eigenvalues(A))
