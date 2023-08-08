import numpy as np
def SLAU_solve_with_LU(B, LU):
    #   Ax=b -> LUx=b -> Ly=b -> Ux=y
    # 1
    # находим столбец y
    y = np.zeros((len(LU[0])))  # заполняем нулями вектор y размером с матрицу
    y[0] = B[0]
    for i in range(1, len(y)):
        s1 = 0
        for p in range(i):
            s1 += LU[i][p] * y[p]
        y[i] = B[i] - s1  # y[2] = b2 - l[2][0]*y[0] - l[2][1]*y[1]

    # 2
    # находим столбец Х
    x = np.zeros((len(LU[0])))
    x[-1] = y[-1] / LU[-1][-1]
    for i in reversed(range(0, len(x) - 1)):
        s2 = 0
        for p in range(i + 1, len(x)):
            s2 += LU[i][p] * x[p]
        x[i] = (y[i] - s2) / LU[i][i]
    return x


def Inverse_A(LU):  # обратная матрица
    # A*A^(-1) = E
    # A*X = E => A^(-1) = X

    # создаем единичную матрицу
    E = np.zeros((len(LU), len(LU)))  # создаем матрицу из нулей
    for i in range(len(E)):  # на диагонали делаем единицы
        E[i][i] = 1.0

    x = []
    for i in range(0, len(LU)):  # находим решение при b = единичная матрица
        x.append(SLAU_solve_with_LU(E[i], LU))

    # полученная матрица будет обратной матрицей А
    x = transpose(
        x)  # транспонируем полученную матрицу, тк вектора х заполнялись в строчку [x1, x2, x3] [x4, x5, x6] а это столбцы!!
    return x


def transpose(matrix):  # транспонирование матрицы
    res = []
    n = len(matrix)
    m = len(matrix[0])
    # проходимся по строчкам и делаем из них столбцы
    for j in range(m):
        tmp = []
        for i in range(n):
            tmp = tmp + [matrix[i][j]]  # склеиваем элементы в строке и превращаем в столбец
        res = res + [tmp]  # добавляем готовые столбцы в res и res уже матрица
    return res


def f_LU(A):
    LU = A
    for i in range(1, len(A)):
        for j in range(i, len(A)):
            LU[j][i - 1] = round(LU[j][i - 1] / LU[i - 1][i - 1], 5)
            prom = [0 for x in range(i)]
            prom.extend([-1 * x * LU[j][i - 1] for x in LU[i - 1][i:]])
            LU[j] = list(map(sum, zip(LU[j], prom)))
            LU[j] = [round(x, 5) for x in LU[j]]
    return LU


def determ(LU):  # определитель матрицы
    determ = 1
    for i in range(len(LU)):
        determ *= LU[i][i]
    return determ


matrixA = [
    [1.0, -5.0, -7.0, 1.0],
    [1.0, -3.0, -9.0, -4.0],
    [-2.0, 4.0, 2.0, 1.0],
    [-9.0, 9.0, 5.0, 3.0]
]

print('   Лабораторная работа №1')
print('     LU - разложение')
print('Кузнецова Дарина М8О-305Б-20')
print('-----------------------------')
print('Ax=b -> LUx=b -> Ly=b -> Ux=y')
print('\n')
print('Матрица A = ')
for i in matrixA:
    print(i)

print("\n")
print("Матрица b: ")
matrixB = [-75, -41, 18, 29]
print(matrixB)

print('\n')
print("Матрица LU = ")
matrLU = f_LU(matrixA)
for k in matrLU:
    print(k)

print('\n')
print('Решение СЛАУ:')
matrX = SLAU_solve_with_LU(matrixB, matrLU)
for i in range(len(matrX)):
    matrX[i] = round(matrX[i])
print(matrX)

print('\n')
print('Обратная матрица')
inv_a = Inverse_A(matrLU)
for k in inv_a:
    print(k)

print('\n')
determ = round(determ(matrLU))
print('Определитель матрицы =', determ)
