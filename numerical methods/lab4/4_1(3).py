import numpy as np
import matplotlib.pyplot as plt


def ddf(x, f, df):
    return ((x + 1) * df / x + 2 * (x - 1) * f / x)


def f(x):
    return (np.exp(2 * x) / (3 * np.exp(2)) + ((3 * x + 1) * np.exp(-1 * x)) / (3 * np.exp(1)))


def Euler(ddy, limits, y0, z0, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    z = np.zeros(N)
    y[0] = y0
    z[0] = z0
    for i in range(0, N - 1):
        z[i + 1] = z[i] + h * ddy(x[i], y[i], z[i])
        y[i + 1] = y[i] + h * z[i]
    return y, z

def Euler_implicit(ddy, limits, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]
    y = RungeKutta(ddy, limits, y0, z0, h)[0]
    z = RungeKutta(ddy, limits, y0, z0, h)[1]
    y[0] = y0
    z[0] = z0
    for i in range(0, N - 1):
        z[i + 1] = z[i] + h * ddy(x[i+1], y[i+1], z[i+1])
        y[i + 1] = y[i] + h * z[i]

    return y

def RungeKutta(ddy, limits, y0, z0, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    z = np.zeros(N)
    y[0] = y0
    z[0] = z0
    if p == 4:
        for i in range(N - 1):
            K1 = h * z[i]
            L1 = h * ddy(x[i], y[i], z[i])
            K2 = h * (z[i] + a[1] * L1)
            L2 = h * ddy(x[i] + b[0] * h, y[i] + b[0] * K1, z[i] + b[0] * L1)
            K3 = h * (z[i] + a[2] * L2)
            L3 = h * ddy(x[i] + b[1] * h, y[i] + b[1] * K2, z[i] + b[1] * L2)
            K4 = h * (z[i] + a[3] * L3)
            L4 = h * ddy(x[i] + b[2] * h, y[i] + b[2] * K3, z[i] + b[2] * L3)
            delta_y = c[0] * K1 + c[1] * K2 + c[2] * K3 + c[3] * K4
            delta_z = c[0] * L1 + c[1] * L2 + c[2] * L3 + c[3] * L4
            y[i + 1] = y[i] + delta_y
            z[i + 1] = z[i] + delta_z
    else:
        for i in range(N - 1):
            K1 = h * z[i]
            L1 = h * ddy(x[i], y[i], z[i])
            K2 = h * (z[i] + b[0] * L1)
            L2 = h * ddy(x[i] + a[1] * h, y[i] + b[0] * K2, z[i] + b[0] * L1)
            delta_y = c[0] * K1 + c[1] * K2
            delta_z = c[0] * L1 + c[1] * L2
            y[i + 1] = y[i] + delta_y
            z[i + 1] = z[i] + delta_z

    return y, z


def Adams(ddy, limits, h):
    x = np.arange(limits[0], limits[1] + h, h)
    N = np.shape(x)[0]
    y = RungeKutta(ddy, limits, y0, z0, h)[0]
    z = RungeKutta(ddy, limits, y0, z0, h)[1]
    if p == 4:
        for i in range(3, N - 1):
            z[i+1] = z[i] + (h / 24) * (55 * ddy(x[i], y[i], z[i]) -
                                        59 * ddy(x[i - 1], y[i - 1], z[i - 1]) +
                                        37 * ddy(x[i - 2], y[i - 2], z[i - 2]) -
                                        9 * ddy(x[i - 3], y[i - 3], z[i - 3]))
            y[i+1] = y[i] + (h / 24) * (55 * z[i] - 59 * z[i - 1] + 37 * z[i - 2] - 9 * z[i - 3])
    if p == 2:
        for i in range(1, N - 1):
            z[i + 1] = z[i] + (h / 2) * (3 * ddy(x[i], y[i], z[i]) - ddy(x[i - 1], y[i - 1], z[i - 1]))
            y[i + 1] = y[i] + (h / 2) * (3 * z[i] - z[i - 1])

    return y


def sqr_error(y, y_correct):
    return np.sqrt(np.sum((y - y_correct) ** 2))


def RungeRomberg(y1, y2, h1, h2, p):
    if h1 > h2:
        k = int(h1 / h2)
        y = np.zeros(np.shape(y1)[0])
        for i in range(np.shape(y1)[0]):
            y[i] = y2[i * k] + (y2[i * k] - y1[i]) / (k ** p - 1)
        return y
    else:
        k = int(h2 / h1)
        y = np.zeros(np.shape(y2)[0])
        for i in range(np.shape(y2)[0]):
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)
        return y


if __name__ == '__main__':
    limits = (1, 2)
    y0 = 0.51378038
    z0 = 0.621555
    h = 0.01

    p = int(input("Введите порядок: "))
    if p == 2:
        a = [0, 2./3]
        b = [2./3]
        c = [0.25, 0.75]
    else:
        a = [0, 0.5, 0.5, 1]
        b = [0.5, 0.5, 1]
        c = [1./6, 1./3, 1./3, 1./6]


    x = np.arange(limits[0], limits[1] + h, h)
    y = f(x)
    y1, z1 = Euler(ddf, limits, y0, z0, h)
    y1_1 = Euler_implicit(ddf, limits, h)
    y2, z = RungeKutta(ddf, limits, y0, z0, h)
    y3 = Adams(ddf, limits, h)

    h2 = h / 2
    y1_2, z_2 = Euler(ddf, limits, y0, z0, h2)
    y_1_2 = y1_2
    y1_1_2 = Euler_implicit(ddf, limits, h2)
    y2_2, z2 = RungeKutta(ddf, limits, y0, z0, h2)
    y3_2 = Adams(ddf, limits, h2)
    print("Оценка погрешности методом Рунге-Ромберга:")
    print("Для явного метода Эйлера:", sqr_error(y1, RungeRomberg(y1, y1_2, h, h2, 1)))
    print("Для неявного метода Эйлера:", sqr_error(y1_1, RungeRomberg(y1_1, y1_1_2, h, h2, 1)))
    print("Для метода Рунге-Кутты:", sqr_error(y2, RungeRomberg(y2, y2_2, h, h2, 4)))
    print("Для метода Адамса:", sqr_error(y3, RungeRomberg(y3, y3_2, h, h2, 4)))

    print("\nСравнение с точным решением:")
    print("Для явного метода Эйлера:", sqr_error(y1, y))
    print("Для неявного метода Эйлера:", sqr_error(y1_1, y))
    print("Для метода Рунге-Кутты:", sqr_error(y2, y))
    print("Для метода Адамса:", sqr_error(y3, y))

    plt.figure(figsize=(12, 7))

    plt.plot(x, y, label='Точное решение', linewidth=1, color="red")
    plt.plot(x, y1, label='Явный метод Эйлера', linewidth=1, color="blue")
    plt.plot(x, y1_1, label='Неявный метод Эйлера', linewidth=1, color="orange")
    plt.plot(x, y3, label='Метод Адамса', linewidth=1, color="black")
    plt.plot(x, y2, label='Метод Рунге-Кутты', linewidth=1, color="yellow")
    plt.grid()
    plt.legend()
    plt.show()
