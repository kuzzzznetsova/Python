import numpy as np, matplotlib.pyplot as plt

def f(x):
    return np.sqrt(x)

def func(x1, x2):
    return (np.sqrt(x1) - np.sqrt(x2)) / (x1 - x2)


def func1(x1, x2, x3):
    return (func(x1, x2) - func(x1, x2)) / (x1 - x2)


def func2(x1, x2, x3, x4):
    return (func1(x1, x2, x3) - func1(x2, x3, x4)) / (x1 - x4)


def func_for_w(x):  #w для многочлена Лагранжа
    w = np.zeros(4)
    w[0] = (x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3])
    w[1] = (x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3])
    w[2] = (x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3])
    w[3] = (x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2])
    return w

def L_method(x, X):
    table = np.zeros((5, 4))
    for i in range(len(x)):
        table[0][i] = x[i]
        table[1][i] = np.sqrt(x[i])
        table[2][i] = func_for_w(x)[i]
        table[3][i] = table[1][i] / table[2][i]
        table[4][i] = X - x[i]

    L = (table[3][0] * (x - table[0][1]) * (x - table[0][2]) * (x - table[0][3])
         + table[3][1] * (x - table[0][0]) * (x - table[0][2]) * (x - table[0][3])
         + table[3][2] * (x - table[0][0]) * (x - table[0][1]) * (x - table[0][3])
         + table[3][3] * (x - table[0][0]) * (x - table[0][1]) * (x - table[0][2]))
    y = np.sqrt(x)
    delta = abs(y - L)
    return L, y, delta


def N_method(x):
    table = np.zeros((5, 4))
    for i in range(len(x)):
        table[0][i] = x[i]
        table[1][i] = np.sqrt(x[i])
        if i != 3:
            table[2][i] = func(x[i], x[i + 1])
        if i < 2:
            table[3][i] = func1(x[i], x[i + 1], x[i + 2])
        table[4][0] = func2(x[0], x[1], x[2], x[3])
    P = (table[1][0] + table[2][0] * (x - table[0][0]) + table[3][0] * (x - table[0][0]) * (x - table[0][1])
         + table[4][0] * (x - table[0][0]) * (x - table[0][1]) * (x - table[0][2]))
    y = np.sqrt(x)
    delta = abs(y - P)
    return P, y, delta




def main():

    x1 = np.array([0, 1.7, 3.4, 5.1])
    x2 = np.array([0, 1.7, 4.0, 5.1])
    root = 3.0

    L1, Ly1, Ldelta1 = L_method(x1, root)
    L2, Ly2, Ldelta2 = L_method(x2, root)
    P1, Py1, Pdelta1 = N_method(x1)
    P2, Py2, Pdelta2 = N_method(x2)

    print('Лабораторная работа №3')
    print('Методы приближения функций. Численное дифференцирование и интегрирование')
    print('Кузнецова Дарина М8О-305Б-20')

    print("Метод Лагранжа:\n"
          "          L(x)                      y(x)      \n",
          'a)', np.around(L1, 2), np.around(Ly1, 2), "\n",
          'б)', np.around(L2, 2), np.around(Ly2, 2))
    print('Погрешность вычисления методом Лагранжа:')
    print('a)', np.around(Ldelta1, 2))
    print('a)', np.around(Ldelta2, 2))
    print("Метод Ньютона:\n"
          "          P(x)                      y(x)        \n",
          np.around(P1, 2), np.around(Py1, 2), "\n",
          np.around(P2, 2), np.around(Py2, 2))
    print('Погрешность вычисления методом Ньютона:')
    print('a)', np.around(Pdelta1, 2))
    print('a)', np.around(Pdelta2, 2))

main()
