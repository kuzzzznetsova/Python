import numpy as np
import matplotlib.pyplot as plt


def dz(x, y, dy):
    return 4 * x * dy - (4 * x ** 2 - 2) * y - 1.2


def solution(x):
    return (1 + x) * np.e ** (x ** 2)


def itter(y0, z0, x, h):

    X0 = np.array([y0, z0])
    m = 0
    while m < 4:
        y1 = y0 + h * X0[1]
        z1 = z0 + h * dz(x, X0[0], X0[1])
        m += 1
        X0 = [y1, z1]
    return y1, z1


def Euler(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y[0] = y0
    z[0] = z0
    for i in range(1, len(x)):
        z[i] = z[i - 1] + h * dz(x[i - 1], y[i - 1], z[i - 1])
        y[i] = y[i - 1] + h * z[i - 1]
    return y


def N_Euler(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y[0] = y0
    z[0] = z0
    for i in range(1, len(x)):
        y[i] = itter(y[i - 1], z[i - 1], x[i], h)[0]
        z[i] = itter(y[i - 1], z[i - 1], x[i], h)[1]
    return y


def RK_2(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y[0] = y0
    z[0] = z0
    K = np.zeros(2)
    L = np.zeros(2)
    for i in range(1, len(x)):
        K[0] = h * z[i - 1]
        L[0] = h * dz(x[i - 1], y[i - 1], z[i - 1])
        K[1] = h * (z[i - 1] + L[0])
        L[1] = h * dz(x[i - 1] + h, y[i - 1] + K[0], z[i - 1] + L[0])
        deltay = (K[0] + K[1]) / 2
        deltaz = (L[0] + L[1]) / 2
        y[i] = y[i - 1] + deltay
        z[i] = z[i - 1] + deltaz
    return y, z


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

    return y, z


def Adams_2(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y_start = RK_4(a, a + h, h, y0, z0)[0]
    for i in range(len(y_start)):
        y[i] = y_start[i]
    z_start = RK_4(a, a + h, h, y0, z0)[1]
    for i in range(len(z_start)):
        z[i] = z_start[i]

    for i in range(2, len(x)):
        z[i] = z[i - 1] + h * (3 * dz(x[i - 1], y[i - 1], z[i - 1]) - dz(x[i - 2], y[i - 2], z[i - 2])) / 2
        y[i] = y[i - 1] + h * z[i - 1]

    return y


def Adams_4(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    z = np.zeros(len(x))
    y_start = RK_4(a, a + 3 * h, h, y0, z0)[0]
    for i in range(len(y_start)):
        y[i] = y_start[i]

    z_start = RK_4(a, a + 3 * h, h, y0, z0)[1]
    for i in range(len(z_start)):
        z[i] = z_start[i]

    for i in range(4, len(x)):
        z[i] = (z[i - 1] + h * (
                55 * dz(x[i - 1], y[i - 1], z[i - 1]) - 59 * dz(x[i - 2], y[i - 2], z[i - 2])
                + 37 * dz(x[i - 3], y[i - 3], z[i - 3]) - 9 * dz(x[i - 4], y[i - 4], z[i - 4])) / 24)
        y[i] = y[i - 1] + h * z[i - 1]

    return y

def RRR(a, b, h, y0, z0):
    x = np.arange(a, b + h, h)
    Euler1 = np.zeros(len(x))
    Euler2 = np.zeros(len(x))
    Runge_Kutty1 = np.zeros(len(x))
    Runge_Kutty2 = np.zeros(len(x))
    Adams1 = np.zeros(len(x))
    Adams2 = np.zeros(len(x))
    N_Euler_norm = N_Euler(a, b, h, y0, z0)
    N_Euler_half = N_Euler(a, b, h / 2, y0, z0)
    Euler_norm = Euler(a, b, h, y0, z0)
    Euler_half = Euler(a, b, h / 2, y0, z0)
    Runge_Kutty2_norm = RK_2(a, b, h, y0, z0)[0]
    Runge_Kutty2_half = RK_2(a, b, h / 2, y0, z0)[0]
    Runge_Kutty_norm = RK_4(a, b, h, y0, z0)[0]
    Runge_Kutty_half = RK_4(a, b, h / 2, y0, z0)[0]
    Adams_norm2 = Adams_2(a, b, h, y0, z0)
    Adams_half2 = Adams_2(a, b, h / 2, y0, z0)
    Adams_norm = Adams_4(a, b, h, y0, z0)
    Adams_half = Adams_4(a, b, h / 2, y0, z0)

    for i in range(len(x)):
        Euler1[i] = Euler_norm[i] + (Euler_half[i * 2] - Euler_norm[i]) / (1 - 0.5 ** 1)
        Euler2[i] = N_Euler_norm[i] + (N_Euler_half[i * 2] - N_Euler_norm[i]) / (1 - 0.5 ** 1)
        Runge_Kutty1[i] = Runge_Kutty_norm[i] + (Runge_Kutty_half[i * 2] - Runge_Kutty_norm[i]) / (1 - 0.5 ** 2)
        Runge_Kutty2[i] = Runge_Kutty2_norm[i] + (Runge_Kutty2_half[i * 2] - Runge_Kutty2_norm[i]) / (1 - 0.5 ** 2)
        Adams1[i] = Adams_norm[i] + (Adams_half[i * 2] - Adams_norm[i]) / (1 - 0.5 ** 2)
        Adams2[i] = Adams_norm2[i] + (Adams_half2[i * 2] - Adams_norm2[i]) / (1 - 0.5 ** 2)

    return Euler1, Runge_Kutty1, Adams1, Euler2, Runge_Kutty2, Adams2


x = np.arange(0, 1.1, 0.1)
y = solution(x)
y1 = Euler(0, 1, 0.1, 1, 1)
y2 = RK_4(0, 1, 0.1, 1, 1)[0]
y3 = Adams_4(0, 1, 0.1, 1, 1)
y4 = N_Euler(0, 1, 0.1, 1, 1)
y5 = Adams_2(0, 1, 0.1, 1, 1)
y6 = RK_2(0, 1, 0.1, 1, 1)[0]
print('   Лабораторная работа №4')
print('Кузнецова Дарина М8О-305Б-20')
print("Методы Эйлера, Рунге-Кутты и Адамса 4-го порядка")
print('\n')
print('Точное решение', np.around(y, 3), '\n')
print('Явный метод Эйлера', np.around(y1, 3))
print('Метода Рунге-Ромберга-Ричардсона ', np.around(RRR(0, 1, 0.1, 1, 1)[0], 3))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[0] - y), 3), '\n')

print('Неявный метод Эйлера', np.around(y4, 3))
print('Метода Рунге-Ромберга-Ричардсона ', np.around(RRR(0, 1, 0.1, 1, 1)[3], 3))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[3] - y), 3), '\n')

print('Метод Рунге-Кутты 2 порядка', np.around(RK_2(0, 1, 0.1, 1, 1)[0], 3))
print('Метода Рунге-Ромберга-Ричардсона ', np.around(RRR(0, 1, 0.1, 1, 1)[4], 5))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[4] - y), 5), '\n')

print('Метод Рунге-Кутты 4 порядка', np.around(RK_4(0, 1, 0.1, 1, 1)[0], 3))
print('Метод Рунге-Ромберга-Ричардсона', np.around(RRR(0, 1, 0.1, 1, 1)[1], 3))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[1] - y), 3), '\n')

print('Метод Адамса 2 порядка', np.around(y5, 3))
print('Метод Рунге-Ромберга-Ричардсона ', np.around(RRR(0, 1, 0.1, 1, 1)[5], 3))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[5] - y), 3), '\n')

print('Метод Адамса 4 порядка', np.around(y3, 3))
print('Метод Рунге-Ромберга-Ричардсона ', np.around(RRR(0, 1, 0.1, 1, 1)[2], 3))
print('Ошибка:', np.around(abs(RRR(0, 1, 0.1, 1, 1)[2] - y), 3), '\n')

plt.figure(figsize=(12, 7))
plt.plot(x, y2, label='Метод Рунге-Кутты 4-го порядка', linewidth=1, color="red")
plt.plot(x, y4, label='Неявный метод Эйлера', linewidth=1, color="orange")
plt.plot(x, y3, label='Метод Адамса 4-го порядка', linewidth=1, color="yellow")
plt.plot(x, y6, label='Метод Рунге-Кутты 2-го порядка', linewidth=1, color="green")
plt.plot(x, y, label='Точное решение', linewidth=1, color="cyan")
plt.plot(x, y5, label='Метод Адамса 2-го порядка', linewidth=1, color="blue")
plt.plot(x, y1, label='Явный метод Эйлера', linewidth=1, color="purple")
plt.grid()
plt.legend()
plt.show()
