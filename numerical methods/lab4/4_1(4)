import numpy as np
import matplotlib.pyplot as plt

def given_solution(x):
    return np.cos(x) + (11/8) * np.sin(x) - np.sin(3*x) / 8

def diff_eq(x, y):
    return np.sin(3*x) - y

y_0 = 1
dy_0 = 1
y_rk = []

def euler_e(h, interval, a, b, c, p):
    x = np.arange(interval[0], interval[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    dy = np.zeros(N)
    y[0] = y_0
    dy[0] = dy_0

    for i in range(0, N - 1):
        dy[i + 1] = dy[i] + h * diff_eq(x[i], y[i])
        y[i + 1] = y[i] + h * dy[i]

    return y

def sekush(func, eps, x_prev, x_cur):
    while (x_cur - x_prev > eps):
        x_new = x_cur - func(x_cur) * (x_cur - x_prev) / \
            (func(x_cur) - func(x_prev))

        x_prev, x_cur = x_cur, x_new

    return x_new

def get_func(y_k, x_kp1):
    def func(y_kp1):
        return y_k + h * diff_eq(x_kp1, y_kp1)

    return func

def euler_i(h, interval, a, b, c, p):
    x = np.arange(interval[0], interval[1] + h, h)
    N = np.shape(x)[0]
    y, dy = runge_kutta(h, interval, a, b, c, p)
    y = np.zeros(N)
    dy = np.zeros(N)
    y[0] = y_0
    dy[0] = dy_0

    for i in range(0, N - 1):
        y[i + 1] = sekush(get_func(y[i], x[i + 1]), 0.001, y[i], i)

        dy[i + 1] = dy[i] + h * diff_eq(x[i + 1], y[i + 1])
        y[i + 1] = y[i] + h * dy[i]

    return y

def runge_kutta(h, borders, a, b, c, p):
    x = np.arange(borders[0], borders[1] + h, h)
    global y_rk
    N = np.shape(x)[0]
    y = np.zeros(N)
    dy = np.zeros(N)
    y[0] = y_0
    dy[0] = dy_0

    if p == 4:
        for i in range(N - 1):
            K1 = h * dy[i]
            L1 = h * diff_eq(x[i], y[i])

            K2 = h * (dy[i] + a[1] * L1)
            L2 = h * diff_eq(x[i] + b[0] * h, y[i] + b[0] * K1)

            K3 = h * (dy[i] + a[2] * L2)
            L3 = h * diff_eq(x[i] + b[1] * h, y[i] + b[1] * K2)

            K4 = h * (dy[i] + a[3] * L3)
            L4 = h * diff_eq(x[i] + b[2] * h, y[i] + b[2] * K3)

            delta_y = c[0] * K1 + c[1] * K2 + c[2] * K3 + c[3] * K4
            delta_z = c[0] * L1 + c[1] * L2 + c[2] * L3 + c[3] * L4

            y[i + 1] = y[i] + delta_y
            dy[i + 1] = dy[i] + delta_z
    else:
        for i in range(N - 1):
            K1 = h * dy[i]
            L1 = h * diff_eq(x[i], y[i])

            K2 = h * (dy[i] + b[0] * L1)
            L2 = h * diff_eq(x[i] + a[1] * h, y[i] + b[0] * K2)

            delta_y = c[0] * K1 + c[1] * K2
            delta_z = c[0] * L1 + c[1] * L2

            y[i + 1] = y[i] + delta_y
            dy[i + 1] = dy[i] + delta_z

    y_rk = y
    return y, dy

def adams(h, interval, a, b, c, p):
    x = np.arange(interval[0], interval[1] + h, h)
    N = np.shape(x)[0]
    y, dy = runge_kutta(h, interval, a, b, c, p)
    if p == 4:
        for i in range(3, N - 1):
            dy[i + 1] = dy[i] + (h / 24) * (55 * diff_eq(x[i], y[i]) -
                                            59 * diff_eq(x[i - 1], y[i - 1]) +
                                            37 * diff_eq(x[i - 2], y[i - 2]) -
                                            9 * diff_eq(x[i - 3], y[i - 3]))

            y[i + 1] = y[i] + (h / 24) * (55 * dy[i] - 59 *
                                          dy[i - 1] + 37 * dy[i - 2] - 9 * dy[i - 3])
    else:
        for i in range(1, N - 1):
            dy[i + 1] = dy[i] + (h / 2) * (3 * diff_eq(x[i],
                                                       y[i]) - diff_eq(x[i - 1], y[i - 1]))

            y[i + 1] = y[i] + (h / 2) * (3 * dy[i] - dy[i - 1])

    return y

def runge_rombert_adjustment(y1, y2, h1, h2, p):
    if h1 > h2:
        h1, h2 = h2, h1
        y1, y2 = y2, y1

    k = int(h2 / h1)
    y = np.zeros(np.shape(y2)[0])

    for i in range(np.shape(y2)[0]):
        y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)

    return y

def root_mse(y, y_correct):
    return np.sqrt(np.sum((y-y_correct)**2))

if __name__ == '__main__':
    print()

    # Шаг, интервал, X
    h = 0.1
    interval = (0., 1.)
    x = np.arange(interval[0], interval[1] + h, h)

    # Точное решение
    y = given_solution(x)

    p = int(input("Введите порядок: "))

    # Таблица Бутчера
    if p == 2:  # метод Хойна
        a = [0, 2./3]
        b = [2./3]
        c = [0.25, 0.75]
    if p == 4:  # классическая схема Рунге-Кутта
        a = [0, 0.5, 0.5, 1]
        b = [0.5, 0.5, 1]
        c = [1./6, 1./3, 1./3, 1./6]

    # Методы Эйлера
    y_e_explicit = euler_i(h, interval, a, b, c, p)
    y_e_implicit = euler_e(h, interval, a, b, c, p)

    print("Решение явным методом Эйлера:\n", y_e_explicit)
    print("Решение неявным методом Эйлера:\n", y_e_implicit)

    print()
    print("Оценка точности:")
    print("  Явный метод Эйлера:", root_mse(y_e_explicit, y))
    print("  Неявный метод Эйлера:", root_mse(y_e_implicit, y))

    # Получаем решения методов для шага h/2
    h2 = h / 2
    y_e_explicit_2 = euler_i(h2, interval, a, b, c, p)
    y_e_implicit_2 = euler_e(h2, interval, a, b, c, p)

    print("\nОценка погрешности методом Рунге-Ромберта:")
    print("  Явный метод Эйлера:",
          root_mse(y_e_implicit, runge_rombert_adjustment(y_e_implicit, y_e_implicit_2, h, h2, p)))
    print("  Неявный метод Эйлера:",
          root_mse(y_e_explicit, runge_rombert_adjustment(y_e_explicit, y_e_explicit_2, h, h2, p)))

    y_runge = runge_kutta(h, interval, a, b, c, p)[0]
    print(f"\nРешение задачи методом Рунге-Кутты {p}-го порядка:\n", y_runge)

    y_adams = adams(h, interval, a, b, c, p)
    print(f"Решение задачи методом Адамса {p}-го порядка:\n", y_adams)

    print("\nСравнение с точным решением:")
    print(f"  Метод Рунге-Кутты {p} порядка:", root_mse(y_runge, y))
    print(f"  Метод Адамса {p} порядка:", root_mse(y_adams, y))

    # для шага h/2
    h2 = h / 2
    y_runge_2 = runge_kutta(h2, interval, a, b, c, p)[0]
    y_adams_2 = adams(h2, interval, a, b, c, p)

    print("\nОценка погрешности методом Рунге-Ромберта при половинном шаге:")
    print("  Метод Рунге-Кутты:", root_mse(
        y_e_explicit, runge_rombert_adjustment(y_e_explicit, y_runge_2, h, h2, p)))
    print("  Метод Адамса:", root_mse(
        y_e_implicit, runge_rombert_adjustment(y_e_implicit, y_adams_2, h, h2, p)))
    print()

    plt.figure(figsize=(9, 5))
    plt.plot(x, y_e_explicit, label='Явный метод Эйлера',
             linewidth=1, color="red")
    plt.plot(x, y_e_implicit, label='Неявный метод Эйлера',
             linewidth=1, color="blue")
    plt.plot(x, y_runge, label='Метод Рунге-Кутты',
             linewidth=1, color="orange")
    plt.plot(x, y_adams, label='Метод Адамса', linewidth=1, color="green")
    plt.plot(x, y, label='Точное решение', linewidth=1, color="black")

    plt.grid()
    plt.legend()
    plt.show()
