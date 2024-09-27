import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sqrt(x)

def fx(X):
    if len(X) == 2:
        return (f(X[0]) - f(X[1])) / (X[0] - X[1])
    else:

        return (fx(X[:len(X) - 1]) - fx(X[1:])) / (X[0] - X[len(X) - 1])

def xx(x, X):
    j = 1
    for i in range(len(X)):
        j *= x - X[i]
    return j

def lagranzh(x, X):
    y = f(X)
    L = 0
    l1 = 1
    l2 = 1
    for i in range(len(X)):
        for j in range(len(X)):
            if (i != j):
                l1 *= (x - X[j])
                l2 *= (X[i] - X[j])
        L += y[i] * l1 / l2
        l1 = 1
        l2 = 1

    return L

def Newton(x, X):
    p = f(X[0]) + fx(X[0:2]) * xx(x, X[0:1]) + xx(x, X[0:2]) * fx(X[0:3]) + xx(x, X[0:3]) * fx(X[0:4])
    return p

X1 = [0, 1.7, 3.4, 5.1]
X2 = [0, 1.7, 4.0, 5.1]
x = np.arange(0, 5.5, 0.01)

plt.figure(figsize=(12, 7))

plt.subplot(1, 2, 1)
plt.scatter(X1, f(X1), color="blue", s=25)
plt.plot(x, f(x), label='f(x)', linewidth=1, color="blue")
plt.plot(x, lagranzh(x, X1), label='Многочлен Лагранжа 3 степени', linewidth=1, color="green")
plt.plot(x, Newton(x, X1), label='Многочлен Ньютона 3 степени', linewidth=1, color="orange")
plt.title("Первый набор точек")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(X2, f(X2), color="blue", s=25)
plt.plot(x, f(x), label='f(x)', linewidth=1, color="blue")
plt.plot(x, lagranzh(x, X2), label='Многочлен Лагранжа 3 степени', linewidth=1, color="green")
plt.plot(x, Newton(x, X2), label='Многочлен Ньютона 3 степени', linewidth=1, color="orange")
plt.title("Второй набор точек")
plt.legend()
plt.grid()
print('   Лабораторная работа №3')
print('Кузнецова Дарина М8О-305Б-20')
print("Многочлены Лагранжа и Ньютона")

print('Значение функции в точке 3.0 = ', f(3.0), '\n')

print('Значение многочлена Лагранжа при наборе X1 в точке 3.0 = ', lagranzh(3.0, X1))
print('Значение многочлена Ньютона при наборе X1 в точке 3.0 = ', Newton(3.0, X1), '\n')
print('Значение многочлена Лагранжа при наборе X2 в точке 3.0 = ', lagranzh(3.0, X2))
print('Значение многочлена Ньютона при наборе X2 в точке 3.0 = ', Newton(3.0, X2), '\n')

print('Погрешность многочлена Лагранжа при наборе X1 = ', abs(f(3.0) - lagranzh(3.0, X1)))
print('Погрешность многочлена Ньютона при наборе X1 = ', abs(f(3.0) - Newton(3.0, X1)), '\n')
print('Погрешность многочлена Лагранжа при наборе X2 = ', abs(f(3.0) - lagranzh(3.0, X2)))
print('Погрешность многочлена Ньютона при наборе X2 = ', abs(f(3.0) - Newton(3.0, X2)), '\n')

plt.show()
