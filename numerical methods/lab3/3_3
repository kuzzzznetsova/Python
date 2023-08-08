import numpy
import numpy as np
from matplotlib import pyplot as plt

def least_squares_1(X, Y):
    a0_1 = len(X)
    a1_1 = 0
    y1 = 0
    for i in range(len(X)):
        a1_1 += X[i]
        y1 += Y[i]
    a0_2 = 0
    a1_2 = 0
    y2 = 0
    for i in range(len(X)):
        a0_2 += X[i]
        a1_2 += X[i]**2
        y2 += (Y[i] * X[i])
    A = numpy.array([[a0_1, a1_1], [a0_2, a1_2]])
    B = numpy.array([y1, y2])
    a_i = numpy.linalg.solve(A, B)

    def calculated_function(x):
        return a_i[0] + a_i[1] * x

    squared_error = 0
    i = 0
    while i < len(X):
        squared_error += (calculated_function(X[i])-Y[i])**2
        i += 1

    print("Многочлен 1 степени:")
    print(a_i[0], "+", a_i[1], "* x")
    print("Сумма квадратов ошибок: ", squared_error, '\n')

    return calculated_function


def least_squares_2(X, Y):
    a0_1 = len(X)
    a1_1 = 0
    a2_1 = 0
    y1 = 0

    for i in range(len(X)):
        a1_1 += X[i]
        a2_1 += X[i]**2
        y1 += Y[i]

    a0_2 = 0
    a1_2 = 0
    a2_2 = 0
    y2 = 0

    for i in range(len(X)):
        a0_2 += X[i]
        a1_2 += X[i] ** 2
        a2_2 += X[i] ** 3
        y2 += (Y[i] * X[i])

    a0_3 = 0
    a1_3 = 0
    a2_3 = 0
    y3 = 0

    for i in range(len(X)):
        a0_3 += X[i] ** 2
        a1_3 += X[i] ** 3
        a2_3 += X[i] ** 4
        y3 += (Y[i] * X[i]**2)

    A = numpy.array([[a0_1, a1_1, a2_1],
                     [a0_2, a1_2, a2_2],
                     [a0_3, a1_3, a2_3]])
    B = numpy.array([y1, y2, y3])
    a_i = numpy.linalg.solve(A, B)

    def calculated_function(x):
        return a_i[0] + a_i[1] * x + a_i[2]*x*x

    squared_error = 0
    i = 0
    while i < len(X):
        squared_error += (calculated_function(X[i]) - Y[i]) ** 2
        i += 1

    print("Многочлен 2 степени:")
    print(a_i[0], "+", a_i[1], "* x +", a_i[2], "*x*x")
    print("Сумма квадратов ошибок: ", squared_error,'\n')

    return calculated_function


def draw(f_x1, f_x2, xi, yi):
    x = np.arange(-1.0, 1.5, 0.01)
    plt.figure(figsize=(8, 5))
    plt.scatter(xi, yi, color="black", s=35)
    plt.plot(x, [f_x1(xk) for xk in x],
             label='Приближающий многочлен 1 степени', linewidth=1, color="orange")
    plt.plot(x, [f_x2(xk) for xk in x], label='Приближающий многочлен 2 степени', linewidth=1,
             color="blue")
    plt.grid()
    plt.legend()
    plt.show()

def main():
    print('   Лабораторная работа №3')
    print('Кузнецова Дарина М8О-305Б-20')
    print("МНК")
    xi = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yi = [1.0, 1.0032, 1.0512, 1.2592, 1.8192, 3.0]

    f_x1 = least_squares_1(xi, yi)
    f_x2 = least_squares_2(xi, yi)

    draw(f_x1, f_x2, xi, yi)
main()

