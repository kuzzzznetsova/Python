import numpy as np
def func(a, b, c, d):
    n = len(a)
    p = [0] * n
    q = [0] * n
    x = [0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
        q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])

    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = q[i] + p[i] * x[i + 1]

    return x


def proverka(a, b, c):
    k = 0
    k2 = 0
    for i in range(len(a)):
        if abs(b[i]) >= abs(c[i]) + abs(a[i]):
            k2 += 1
            if abs(b[i]) > abs(c[i]) + abs(a[i]):
                k = k + 1
    if k2 == len(a) and k >= 1:
        return "Критерий |b[i]| >= |a[i]| + |c[i]| выполняется"
    else:
        return "Критерий не выполняется"


a = [0, 2, 4, -3, 3]
c = [8, 4, 5, -7, 0]
b = [15, 15, 11, 16, 8]
f = [92, -84, -77, 15, -11]
n = 5

# A = [[15, 8, 0, 0, 0], [0, 15, 4, 0, 0], [0, 4, 11, 11, 0], [0, 0, -3, 16, -7], [0, 0, 0, 3, 8]]
print('   Лабораторная работа №1')
print('     Метод прогонки')
print('Кузнецова Дарина М8О-305Б-20')
print('-----------------------------')
print('Дана трёхдиагональная матрица вида:')
print(' b1 c1  0   0   0')
print(' 0  a3  b3  c3  0')
print(' 0  0   a4  b4 c4')
print(' 0  0   a4  b4 c4')
print(' 0  0    0  a5 b5')
print('\n')

print(proverka(a, b, c))
print('\n')
print('Решение СЛАУ:')
x = func(a, b, c, f)
print(x)
