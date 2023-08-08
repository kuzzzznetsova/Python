from random import uniform
import matplotlib.pyplot as plt
import math
import numpy as np


xkk = .99999
# # def simple_iterations_method(q, eps):
#  x_0 = -0.6

#     min = -0.8
#     max = -0.4
#     # Проверка выполнения достаточного условия для метода простых итераций

#     phi(min) = -math.sqrt(math.log(min + 2))
#     phi(max) = -math.sqrt(math.log(max + 2))

#     phi_Xx_a = - 1 / ((2*min + 4) * math.sqrt(math.log(min + 2)))
#     phi_Xx_b = - 1 / ((2*max + 4) * math.sqrt(math.log(max + 2)))

#     print("")
#     print("Метод простых итераций")

#     if (min <= phi(min) <= max) and (min <= phi(max) <= max):
#         print("Phi(x) принадлежит заданному промежутку (усл 1)")

#         # Phi'(x) <= q < 1
#         if (abs(phi_Xx_a) <= 1) and abs(phi_Xx_b) <= 1:
#             print("Условия теоремы 2.3 выполнены")
#         else:
#             print("Условия теоремы 2.3 не выполнены")
#             return

#     # Метод итераций
#     x_prev = 0
#     iterations = 0
#     x_k = x_0

#     # Условие сходимости
#     while (q / (1 - q) * abs(x_k - x_prev) > eps):
#         phi_x_k = -math.sqrt(math.log(x_k + 2))

#         x_prev = x_k
#         x_k = phi_x_k
#         iterations = iterations + 1

#     print('Итераций: ', iterations)

#     print('x =', xkk)


#

def phi(x):
    return -math.sqrt(math.log(x + 2))


def d_phi(x):
    return - 1 / ((2*x + 4) * math.sqrt(math.log(x + 2)))


def function(x):
    return math.log(x + 2) - x**2


def dd_function(x):
    return (2*x**2 + 8*x + 9) / (x**2 + 4*x + 4)


def d_function(x):
    return -(2*x**2 + 8*x + 9) / (x**2 + 4*x + 4)


def simple_iterations_method(q, eps):
    x_0 = -0.6

    a = -0.8
    b = -0.4

    print("")
    print("Метод простых итераций")

    # Проверка выполнения достаточного условия для метода простых итераций
    if (a <= phi(a) <= b) and (a <= phi(b) <= b):
        print("Phi(x) принадлежит заданному промежутку (усл 1)")

        # Phi'(x) <= q < 1
        if (abs(d_phi(a)) <= 1) and abs(d_phi(b)) <= 1:
            print("Условия теоремы 2.3 выполнены")
        else:
            return print("Условия теоремы 2.3 не выполнены")

    # Метод итераций
    x_prev = 0
    iterations = 0
    x_k = x_0

    # Условие сходимости
    while (q / (1 - q) * abs(x_k - x_prev) > eps):
        phi_x_k = -math.sqrt(math.log(x_k + 2))

        x_prev = x_k
        x_k = phi_x_k
        iterations = iterations + 1

    print('Итераций: ', iterations)

    print('x =', xkk)


def newton_method(eps, a, b):
    global xkk
    print("")
    print("Метод Ньютона")

    x_0 = None

    if function(a) * function(b) < 0:
        if function(a) * dd_function(a) > 0:
            print("Выполняются условия теоремы 2.2")
            x_0 = a

        elif function(b) * dd_function(b) > 0:
            print("Выполняются условия теоремы 2.2")
            x_0 = b

        else:
            return print("Не выполняются условия теоремы 2.2")
    else:
        return print("Не выполняются условия теоремы 2.2")

    x_prev = -1
    x_k = x_0
    iterations = 0

    while (abs(x_k - x_prev) > eps):
        iterations = iterations + 1

        x_prev = x_k
        x_k = x_prev - (function(x_k)/d_function(x_k))

    xkk *= x_k
    print('Итераций: ', iterations)
    print('X(*):\t', x_k)


def plot(min, max, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    x = np.linspace(min, max, 100)

    y = np.log(x + 2)
    y1 = np.power(x, 2)

    plt.grid()
    ax.plot(x, y)
    ax.plot(x, y1)
    plt.show()


if __name__ == '__main__':
    plot(-4, 4, 'График F1(x) и F2(x)')

    a = 0
    b = 2

    eps = 0.001
    q = 0.99

    is_seidel = False

    newton_method(eps, a, b)

    simple_iterations_method(q, eps)
