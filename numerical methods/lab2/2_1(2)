import matplotlib.pyplot as plt
import math
import numpy as np


def f(x):
    return math.log(x + 1) - x*2 + 0.5

def d_f(x):
    return 1 / (x + 1) - 2

def dd_f(x):
    return 1 / (x + 1)**2

def phi(x):
    return (np.log(x + 1) + 0.5) / 2


def d_phi(x):
    return - 1 / (2*x + 2)



#проверка условия сходимости
def x0(a, b):
    if f(a) * f(b) < 0:
        if f(a) * dd_f(a) > 0:
            print("Условие сходимости выполнено")
            return a

        elif f(b) * dd_f(b) > 0:
            print("Условие сходимости выполнено")
            return b

        else:
            return print("Условие сходимости не выполнено")
    else:
        return print("Условие сходимости не выполнено")

#метод Ньютона
def newton(eps, a, b):
    print("")
    print("Метод Ньютона")

    x_0 = x0(a, b)

    x_k = x_0
    x_k_next = x_k - (f(x_k)/d_f(x_k))
    count = 1

    while (abs(x_k_next - x_k) > eps):
        count = count + 1

        x_k = x_k_next
        x_k_next = x_k - (f(x_k)/d_f(x_k))

    print('Потребовалось итераций: ', count)
    print('X(*):\t', x_k_next)

#Метод простых итераций
def simple_iterations(a, b, q, eps):
    print("")
    print("Метод простых итераций")

    if (a <= phi(a) <= b) and (a <= phi(b) <= b):
        #Phi(x) принадлежит [a, b]

        # Phi'(x) <= q < 1
        if (abs(d_phi(a)) < 1) and (abs(d_phi(b)) < 1):
            x_0 = a
            print("Достаточные условия выполнены")
        else:
            print("Достаточные условия не выполнены")
    else:
        print("Достаточные условия не выполнены")


    x_k = x_0
    iterations = 1
    x_k_next = phi(x_k)

    #оценка погрешности
    while (q / (1 - q) * abs(x_k_next - x_k) > eps):

        x_k = x_k_next
        x_k_next = phi(x_k)
        iterations = iterations + 1

    print('Потребовалось итераций: ', iterations)

    print('x =', x_k_next)

#построение графика
def plot(min, max):
    fig, ax = plt.subplots()
    ax.set_title('f1(x) = ln(x+1);  f2(x) = 2x - 0.5')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    x = np.linspace(min, max, 100)

    y = np.log(x + 1)
    y1 = 2*x - 0.5

    plt.grid()
    ax.plot(x, y)
    ax.plot(x, y1)
    plt.show()




a = -1
b = 0

eps = 0.001
q = 0.99

plot(-1, 4)

newton(eps, a, b)

simple_iterations(a, b, q, eps)
