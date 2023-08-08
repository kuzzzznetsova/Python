import math
import numpy as np, matplotlib.pyplot as plt
def func(b):
    if b == 2: b = 3
    return b

def plot_function_f(x):
    return np.power(2, x), 2 - x * x

def function_f(x):
    return np.power(2, x) + x * x - 2

# производная функции ф
def derivative_f(x):
    return math.log(2) * np.power(2, x) + 2 * x

def second_der_f(x):
    return math.log(2) * math.log(2) * np.power(2, x) + 2

def function_fi(x):
    return ((2 - 2 ** x) * x) ** (1. / 3.)

def derivative_fi(x):
    return (2 - math.log(2) * 2 ** x * x - 2) / (3 * ((2 * x - 2 ** x * x) ** (2. / 3.)))


def iter(a, b, e, q):
    print("1. МЕТОД ПРОСТЫХ ИТЕРАЦИЙ")
    a = 0.4
    b = 0.8
    f_a = function_f(a)
    f_b = function_f(b)
    if float(f_a) * float(f_b) < 0:
        # Проверка выполнения достаточного условия для метода простых итераций
        if (a <= function_fi(a) <= b) and (a <= function_fi(b) <= b):

            print("Phi(x) принадлежит заданному промежутку (усл 1)")

            # Phi'(x) <= q < 1
            if (abs(derivative_fi(a)) <= 1) and (abs(derivative_fi(b)) <= 1):
                print("Условия теоремы 2.3 выполнены" )
            else:
                return print("Условия теоремы 2.3 не выполнены")
        k = 0
        x_0 = a
        x_1 = function_fi(x_0)
        if abs(derivative_fi(x_0)) < 1:
            # критерий окончания
            while q / (1 - q) * abs(x_1 - x_0) >= e:
                x_0 = x_1
                x_1 = function_fi(x_0)
                k = k + 1
            print("Найдено решение:", (np.around(x_1, 5)).real)
            print("Количество итераций =", k)
            print("\n")
    else:
        if a < 0:
            print("Введите положительные границы для корня")
        else:
            print("Корня на данном участке нет")

def newton(a, b, e):
    f_a = function_f(a)
    f_b = function_f(b)
    print("2. МЕТОД НЬЮТОНА")


    if float(f_a) * float(f_b) < 0:

        if function_f(a) * second_der_f(a) > 0:
            print("Выполняются условия теоремы 2.2")
            x_0 = a

        elif function_f(b) * second_der_f(b) > 0:
            print("Выполняются условия теоремы 2.2")
            x_0 = b

        else:
            return print("Не выполняются условия теоремы 2.2")

        k = 0
        x_0 = a
        x_1 = x_0 - function_f(x_0) / derivative_f(x_0)
        if function_f(x_1) * derivative_f(x_1) > 0:
            while abs(x_1 - x_0) >= e:
                k = k + 1
                x_0 = x_1
                x_1 = x_0 - function_f(x_0) / derivative_f(x_0)

                if abs(x_1 - x_0) <= e:
                    return x_1, k
    else:
        if a < 0:
            print("Введите положительные границы для корня")
        else:
            print("Корня на данном участке нет")

def main():
    e = 0.001
    q = 0.99
    plt.title("График степенной функции и параболы")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.axis([-3, 3, -3, 3])
    x = np.arange(-3, 3, 0.1)
    y = plot_function_f(x)
    plt.plot(x, y[0], x, y[1])
    plt.show()
    print('   Лабораторная работа №2')
    print('Метод простой итерации и метод Ньютона для решения нелинейных уравнений')
    print('Кузнецова Дарина М8О-305Б-20')
    print("Введите границы отрезка, на котором производится поиск корня:")
    a = float(input())
    b = float(input())
    if a >= 0:
        print("Корнем уравнения на промежутке от ", a, " до ", b, "является: ")
        print('\n')
        b = func(b)
        iter(a, b, e, q)
        metod_n = newton(a, b, e)
        print("Найдено решение:", np.around(metod_n[0], 5))
        print("Количество итераций =", metod_n[1])
    else:
        print("Введите положительные границы для поиска корня")
main()
