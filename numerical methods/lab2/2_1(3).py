import matplotlib.pyplot as plt
import numpy as np
import math

def y(x):
    return np.exp(x) - 2 * x - 2

def dy(x):
    return np.exp(x) - 2

def d2y(x):
    return np.exp(x)

def fi(x):
    return np.log(2 * x + 2)

def dfi(x):
    return 1 / (1 + x)

def check_cond(x):
    if (y(x) * d2y(x) <= dy(x) ** 2):
        return 0
    else:
        return 1

def select(a, b):
    if (check_cond(a) == 0) and (check_cond(b) == 0):
        if (y(a) * d2y(a) > 0):
            return a
        elif (y(b) * d2y(b) > 0):
            return b
    elif (check_cond(a) == 0):
        return a
    elif (check_cond(b) == 0):
        return b
    else:
        print('невозможно выбрать x0 \n')


def method_it(a, b, q):
    if (abs(dfi(a)) < 1):
        x0 = a
    elif (abs(dfi(b)) < 1):
        x0 = b
    else:
        print('нельзя выбрать корень')
        return 0
    m = 1
    e = 0.001
    n = 0
    while (m > e):
        x = fi(x0)
        m = q / (1 - q) * np.abs(x - x0)
        x0 = x
        n += 1
    print('Корень уравнения x =', x)
    print("Количество итераций =", n)


def solution(a, b):
    e = 0.001
    x0 = select(a, b)
    if (check_cond(x0) == 1):
        return ('не выполняется условие сходимости')
    else:
        print("Выполняется условие сходимости")
    k = 0

    while (x0 != 0.0001):
        xn = x0 - y(x0) / dy(x0)
        if (abs(xn - x0) < e):
            x0 = 0.0001
        else:
            x0 = xn
        k += 1
    print('Корень уравнения x =', xn)
    print("Количество итераций =", k)
    print('\n')

plt.title("f1(x) = 2x + 2, f2(x) = e ** x")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.axis([-2, 6, -2, 6])
x = np.arange(-2, 6, 0.01)
x1 = np.arange(-2, 6, 0.01)
plt.plot(x, np.e ** x )
plt.plot(x1, 2 * x + 2)
plt.show()

print('   Лабораторная работа №2')
print('Метод простой итерации и метод Ньютона для решения нелинейных уравнений')
print('Кузнецова Дарина М8О-305Б-20')
print("Введите границы отрезка, на котором производится поиск корня:")
a2 = float(input())
b2 = float(input())

if (y(a2) * y(b2) < 0):
    print('1.МЕТОД НЬЮТОНА')
    solution(a2, b2)
    print('2.МЕТОД ПРОСТЫХ ИТЕРАЦИЙ')
    method_it(a2, b2, 0.8)
else:
    print('нельзя найти корень')
