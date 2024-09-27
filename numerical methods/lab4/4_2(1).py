import numpy as np
import matplotlib.pyplot as plt
from lab11 import LU_decompose, solving


def diff(x, y, z):
    return -4 * x * z - (4 * x ** 2 + 2) * y


def f(x):
    return (1 + x) * np.exp(-x ** 2)


def p(x):
    return 4 * x


def q(x):
    return 4 * x ** 2 + 2


def right_f(x):
    return 0


def diff_left(BCondition, h):
    return (BCondition['c']-(BCondition['b']/h)*f(h)) / (BCondition['a']-(BCondition['b']/h))


def diff_right(BCondition, h, y):
    return (BCondition['c']+(BCondition['b']/h)*y[-2]) / (BCondition['a']+(BCondition['b']/h))


def ShotingMethod(ddy, limits, BCondition1, BCondition2, h):
    y0 = diff_left(BCondition1, h)

    eta1 = 0.5
    eta2 = 2.0
    resolve1 = RungeKutta(ddy, limits, y0, eta1, h)[0]
    resolve2 = RungeKutta(ddy, limits, y0, eta2, h)[0]
    Phi1 = resolve1[-1] - diff_right(BCondition2, h, resolve1)
    Phi2 = resolve2[-1] - diff_right(BCondition2, h, resolve2)
    while abs(Phi2 - Phi1) > h/10:
        temp = eta2
        eta2 = eta2 - (eta2 - eta1) / (Phi2 - Phi1) * Phi2
        eta1 = temp
        resolve1 = RungeKutta(ddy, limits, y0, eta1, h)[0]
        resolve2 = RungeKutta(ddy, limits, y0, eta2, h)[0]
        Phi1 = resolve1[-1] - diff_right(BCondition2, h, resolve1)
        Phi2 = resolve2[-1] - diff_right(BCondition2, h, resolve2)
    return RungeKutta(ddy, limits, y0, eta2, h)


def Solve(A, b):
    Lu = LU_decompose(A)
    x = solving(Lu, b)
    return x


def FiniteDifferenceMethod(BCondition1, BCondition2, equation, limits, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]

    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0][0] = -2 + h**2 * equation['q'](x[1])
    A[0][1] = 1 + equation['p'](x[1]) * h / 2
    b[0] = -(1 - (equation['p'](x[1]) * h) / 2) * BCondition1['c']
    for i in range(1, N-1):
        A[i][i-1] = 1 - equation['p'](x[i]) * h/2
        A[i][i] = -2 + equation['q'](x[i]) * h**2
        A[i][i+1] = 1 + equation['p'](x[i])*h/2
        b[i] = 0
    A[N-1][N-2] = -BCondition2['b']/h
    A[N-1][N-1] = BCondition2['a'] + BCondition2['b']/h
    b[N-1] = BCondition2['c']
    return Solve(A, b)


def RungeKutta(func, limits, y0, z0, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    z = np.zeros(N)
    y[0] = y0
    z[0] = z0
    for i in range(N-1):
        K1 = h * z[i]
        L1 = h * func(x[i], y[i], z[i])
        K2 = h * (z[i] + 0.5 * L1)
        L2 = h * func(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * (z[i] + 0.5 * L2)
        L3 = h * func(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * (z[i] + L3)
        L4 = h * func(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y[i+1] = y[i] + delta_y
        z[i+1] = z[i] + delta_z

    return y, z


def RungeRomberg(y1, y2, h1, h2, p):
    if h1 > h2:
        k = int(h1 / h2)
        y = np.zeros(np.shape(y1)[0])
        for i in range(np.shape(y1)[0]):
            y[i] = y2[i*k]+(y2[i*k]-y1[i])/(k**p-1)

        return y
    else:
        k = int(h2 / h1)
        y = np.zeros(np.shape(y2)[0])
        for i in range(np.shape(y2)[0]):
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)

        return y


if __name__ == '__main__':
    equation = {'p': p, 'q': q, 'f': right_f}
    BCondition1 = {'a': 0, 'b': 1, 'c': 1}
    BCondition2 = {'a': 4, 'b': -1, 'c': 23 * (np.e ** (-4))}
    limits = (0, 2)
    h = 0.01

    x = np.arange(limits[0], limits[1]+h, h)
    y = f(x)
    y1 = ShotingMethod(diff, limits, BCondition1, BCondition2, h)
    y2 = FiniteDifferenceMethod(BCondition1, BCondition2, equation, limits, h)

    h2 = h / 2
    y1_2 = ShotingMethod(diff, limits, BCondition1, BCondition2, h2)
    y2_2 = FiniteDifferenceMethod(BCondition1, BCondition2, equation, limits, h2)

    plt.plot(x, y, label='Точное решение', linewidth=1, color="blue")
    plt.plot(x, y1[0], label='Метод стрельбы', linewidth=1, color="red")
    plt.plot(x, y2, label='Метод конечных разностей', linewidth=1, color="green")
    plt.grid()
    plt.legend()
    plt.show()
