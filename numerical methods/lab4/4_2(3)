import matplotlib.pyplot as plt
import math
import numpy as np
from functools import reduce

def q_x(x):
    return -2 / (x*x*(x+1))

def p_x(x):
    return 0

def f_x(x):
    return (2 - 2*x) / (x*x*(x+1))

def y_x(x):
    return -1 + (2 + 2*(x + 1) * math.log(abs(x + 1))) / x

def f_xyz(x, y, z):
    return (2*y - 2*x + 2) / (x*x*(x+1))

def epsilon(x, y):
    norm = 0.0

    for i in range(len(x)):
        norm += (y[i] - y_x(x[i]))**2

    return norm**0.5

def get_x(a, b, h):
    return list(np.arange(a, b + h/2, h))

def Prod(items=None):
    return reduce(lambda x, y: x*y, items)

def product_from_numbers(x, arr):
    result = 1

    for number in arr:
        result *= x - number

    return result

def product_from_callable(x, fn_arr):
    result = 1

    for fn in fn_arr:
        if callable(fn):
            result *= fn(x)
        else:
            result *= fn

    return result

def with_product_numbers(arr):
    def hof(x):
        return product_from_numbers(x, arr)

    return hof

def with_product_callable(arr):
    def hof(x):
        return product_from_callable(x, arr)

    return hof

def with_plus(functions):
    def hof(x):
        result = 0
        for fn in functions:
            if callable(fn):
                result += fn(x)
            else:
                result += fn
        return result
    return hof

def build_lagrange(X, Y):
    def l_polynom(_x):
        return 0
    for i in range(len(X)):
        x = X[i]
        iterable = list(X)
        iterable.remove(x)
        f_koeff = Y[i]

        multi_omega_fn = with_product_numbers(iterable)
        omega = multi_omega_fn(x)

        l_polynom = with_plus([
            l_polynom,
            with_product_callable([f_koeff/omega, multi_omega_fn])
        ])

    return l_polynom

def race_method(A, b):
    P = [-item[2] for item in A]
    Q = [item for item in b]

    P[0] /= A[0][1]
    Q[0] /= A[0][1]

    for i in range(1, len(b)):
        z = (A[i][1] + A[i][0] * P[i - 1])
        P[i] /= z
        Q[i] -= A[i][0] * Q[i - 1]
        Q[i] /= z

    x = [item for item in Q]

    for i in range(len(x) - 2, -1, -1):
        x[i] += P[i] * x[i + 1]

    return x

def runge_kutta_4_or(x, y0, z0, h, f=f_xyz):
    y = [y0]
    z = [z0]

    def delta_4p(xk, yk, zk, h, f):
        K1 = h * zk
        L1 = h * f(xk, yk, zk)

        K2 = h * (zk + L1 / 2)
        L2 = h * f(xk + h / 2, yk + K1 / 2, zk + L1 / 2)

        K3 = h * (zk + L2 / 2)
        L3 = h * f(xk + h / 2, yk + K2 / 2, zk + L2 / 2)

        K4 = h * (zk + L3)
        L4 = h * f(xk + h, yk + K3, zk + L3)

        return ((K1 + 2 * K2 + 2 * K3 + K4) / 6, (L1 + 2 * L2 + 2 * L3 + L4) / 6)

    for k in range(len(x) - 1):
        delta = delta_4p(x[k], y[k], z[k], h, f)
        y.append(y[k] + delta[0])
        z.append(z[k] + delta[1])

    return y

def shoot_method(x, y0, y1, h, f=f_xyz, e=0.00001):
    nu_last = 1
    nu_now = 0.8
    y_last = runge_kutta_4_or(x, y0, nu_last, h, f)
    y_now = runge_kutta_4_or(x, y0, nu_now, h, f)
    F_last = y_last[-1] - y1
    F_now = y_now[-1] - y1

    def next_step(F0, F1, nu0, nu1):
        return nu1 - F1 * (nu1 - nu0) / (F1 - F0)

    while abs(F_now) > e:
        nu_last, nu_now = nu_now, next_step(F_last, F_now, nu_last, nu_now)
        y_last, y_now = y_now, runge_kutta_4_or(x, y0, nu_now, h, f)
        F_last, F_now = F_now, y_now[-1] - y1

    return y_now

def get_A(h, p, q, x):
    A = [[1 - (p(x[i]))/2, (-2 + h*h*q(x[i])), 1 + (p(x[i])*h)/2]
         for i in range(1, len(x[:-1]))]
    A[0][0] = 0
    A[-1][-1] = 0

    return A

def get_B(h, p, f, x, y0, y1):
    b = [h*h*f(x[i]) for i in range(1, len(x[:-1]))]
    b[0] -= y0*(1 - p(x[1])*h/2)
    b[-1] -= y1*(1 + p(x[-2])*h/2)

    return b

def difference_method(x, y0, y1, h, p=p_x, q=q_x, f=f_x):
    A = get_A(h, p, q, x)
    b = get_B(h, p, f, x, y0, y1)
    y = [y0] + race_method(A, b) + [y1]

    return y

def runge_romberg(y1, y2, p):
    norm = 0.0

    for i in range(len(y2)):
        norm += (y1[i*2] - y2[i])**2

    return (norm**0.5) / (2**p - 1)

def check_diagonal_dominance(h, x):
    for k in range(len(x)):
        if not (abs(-2 + h**2 * q_x(x[k])) > abs(1 - (p_x(x[k])*h/2)) + abs(1 + (p_x(x[k])*h/2))):
            return False

    return True

if __name__ == '__main__':
    print('   Лабораторная работа №4')
    print('Кузнецова Дарина М8О-305Б-20')
    h = 0.25
    a = 1
    b = 2
    y0 = 1 + 4 * math.log(2)
    y1 = 3 * math.log(3)

    x = get_x(a, b, h)
    xi = get_x(a, b, 0.0005)

    yp1 = shoot_method(x, y0, y1, h)
    Lagrange_shoot = build_lagrange(x, yp1)
    y_shoot = list(map(Lagrange_shoot, xi))

    yp2 = difference_method(x, y0, y1, h)
    Lagrange_diff = build_lagrange(x, yp2)
    y_diff = list(map(Lagrange_diff, xi))

    yt = list(map(y_x, xi))

    # Вычисление погрешности

    y = shoot_method(x, y0, y1, h)

    print()
    if not check_diagonal_dominance(h, x):
        print("Не выполняется условие диагонального преобладания")
    else:
        print("Выполняется условие диагонального преобладания")

    print()
    print(" Метод стрельбы")
    print("Погрешность вычисления:", epsilon(x, y))
    p = 4

    xr = get_x(a, b, h / 2)
    yr = shoot_method(xr, y0, y1, h / 2)
    print("Погрешность вычисления (Рунге-Ромберга-Ричардсон):",
          runge_romberg(yr, y, p))

    print()
    print(" Конечно-разностный метод")
    y = difference_method(x, y0, y1, h)
    print("Погрешность вычисления:", epsilon(x, y))
    p = 2
    yr = difference_method(xr, y0, y1, h / 2)
    print("Погрешность вычисления (Рунге-Ромберга-Ричардсон):",
          runge_romberg(yr, y, p))

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)

    line1, = ax1.plot(xi, y_shoot, 'purple')
    line2, = ax1.plot(xi, y_diff, 'black')
    line3, = ax1.plot(xi, yt, 'red')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.set_xlim(0.97, 2.03)

    ax1.legend((line1, line2, line3),
               ("Метод стрельбы", "Конечно-разностный метод", "Точное решение"))
    ax1.grid()
    plt.show()

