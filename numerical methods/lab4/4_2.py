import numpy as np
import math
import matplotlib.pyplot as plt

def p(x):
    return -math.tan(x)

def q():
    return 2

def solution(x):
    return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))

def dz(x, y, z):
    return math.tan(x) * z - 2 * y


def finite_difference_method(a, b, h, y0, y1):
    x = np.arange(a, b + h, h)
    N = int((b - a) / h)
    A = []
    B = []
    C = []
    D = []
    A.append(0)
    B.append(-2 + h * h * q())
    C.append(1 + p(x[1]) * h / 2)
    D.append(-(1 - (p(x[1]) * h) / 2) * y0)
    for i in range(2, N):
        A.append(1 - p(x[i]) * h / 2)
        B.append(-2 + h * h * q())
        C.append(1 + p(x[i]) * h / 2)
        D.append(0)
    A.append(1 - p(x[N - 2]) * h / 2)
    B.append(-2 + h * h * q())
    C.append(0)
    D.append(-(1 + (p(x[N - 2]) * h) / 2) * y1)

    P = np.zeros(N)
    Q = np.zeros(N)
    P[0] = (-C[0] / B[0])
    Q[0] = (D[0] / B[0])
    for i in range(1, N):
        P[i] = (-C[i] / (B[i] + A[i] * P[i - 1]))
        Q[i] = ((D[i] - A[i] * Q[i - 1]) / (B[i] + A[i] * P[i - 1]))
    ans = np.zeros(N)
    ans[N - 1] = Q[N - 1]
    for i in range(N - 2, 0, -1):
        ans[i] = P[i] * ans[i + 1] + Q[i]
    ans[0] = y0
    ans = np.append(ans, y1)
    return ans


def RK_4(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y[0] = y0
    z[0] = z0
    K = np.zeros(4)
    L = np.zeros(4)
    for i in range(1, len(x)):
        K[0] = h * z[i - 1]
        L[0] = h * dz(x[i - 1], y[i - 1], z[i - 1])

        for j in range(1, 3):
            K[j] = h * (z[i - 1] + L[j - 1] / 2)
            L[j] = h * dz(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2, z[i - 1] + L[j - 1] / 2)
        K[3] = h * (z[i - 1] + L[2])
        L[3] = h * dz(x[i - 1] + h, y[i - 1] + K[2], z[i - 1] + L[2])
        deltay = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6
        deltaz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6
        y[i] = y[i - 1] + deltay
        z[i] = z[i - 1] + deltaz

    return y


def shooting(a, b, h, y0, z0):
    n0 = -10.0
    n1 = 21.0
    f_0 = RK_4(a, b, h, y0, n0)[-1] - z0
    f_1 = RK_4(a, b, h, y0, n1)[-1] - z0
    e = 0.001

    while (f_1 > e):
        n2 = n1 - (n1 - n0) / (f_1 - f_0) * f_1
        n0 = n1
        n1 = n2

        f_0 = RK_4(a, b, h, y0, n0)[-1] - z0
        f_1 = RK_4(a, b, h, y0, n1)[-1] - z0

    return RK_4(a, b, h, y0, n1)


def RRR(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    shooting1 = np.zeros(len(x))
    FD = np.zeros(len(x))
    shooting_norm = shooting(a, b, h, y0, z0)
    shooting_half = shooting(a, b, h / 2, y0, z0)
    FD_norm = finite_difference_method(a, b, h, y0, z0)
    FD_half = finite_difference_method(a, b, h / 2, y0, z0)
    for i in range(len(x)):
        shooting1[i] = shooting_norm[i] + (shooting_half[i * 2] - shooting_norm[i]) / (1 - 0.5 ** 1)
        FD[i] = FD_norm[i] + (FD_half[i * 2] - FD_norm[i]) / (1 - 0.5 ** 1)
    return shooting1, FD


x = np.arange(0.0, math.pi / 6 + math.pi / 60, math.pi / 60)
y = solution(x)
print('Точное решение:', np.around(y, 3))
y0 = 2.0
h = np.pi / 60
a = 0.0
b = np.pi / 6
yb = 2.5 - 0.5 * np.log(3.0)

y22 = shooting(a, b, h, y0, yb)
y33 = finite_difference_method(a, b, h, y0, yb)
plt.figure(figsize=(12, 7))
plt.plot(x, y, label='точное решение', linewidth=1, color="red")
plt.plot(x, y22, label='метод стрельбы', linewidth=1, color="blue")
plt.plot(x, y33, label='конечно-разностный метод', linewidth=1, color="green")
print('Решение методом стрельбы:', np.around(y22, 3))
print('Погрешность метода стрельбы:', np.around(abs(RRR(a, b, h, y0, yb)[0] - y), 6))
print('Решение конечно-разностным методом:', np.around(y33, 3))
print('Погрешность конечно-разностного метода:', np.around(abs(RRR(a, b, h, y0, yb)[1] - y), 4))
plt.ylim(0, 4)
plt.grid()
plt.legend()
#plt.show()
