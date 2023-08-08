import numpy as np

def numerical_differentiation1(x, y, a):
    left_dx = (y[a] - y[a - 1]) / (x[a] - x[a - 1])
    right_dx = (y[a + 1] - y[a]) / (x[a + 1] - x[a])
    dx = ((y[a] - y[a - 1]) / (x[a] - x[a - 1]) +
          (((y[a + 1] - y[a]) / (x[a + 1] - x[a]) - (y[a] - y[a - 1]) / (x[a] - x[a - 1])) / (x[a + 1] - x[a - 1])) *
          (2 * x[a] - x[a - 1] - x[a]))
    return left_dx, right_dx, dx

def numerical_differentiation2(x, y, a):
    dx2 = 2 * (((y[a + 1] - y[a]) / (x[a + 1] - x[a])) - ((y[a] - y[a - 1]) / (x[a] - x[a - 1]))) / (
                x[a + 1] - x[a - 1])
    return dx2

def main(x, y, a):
    print('   Лабораторная работа №3')
    print('Кузнецова Дарина М8О-305Б-20')

    l_dx, r_dx, dx = numerical_differentiation1(x, y, a)
    dx2 = numerical_differentiation2(x, y, a)
    print("Производная со вторым порядком точности: ", np.around(dx, 2))
    print("Вторая производная: ", np.around(dx2, 2))

x = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
y = np.array([1.7722, 1.5708, 1.3694, 1.1593, 0.9273])
a = 2
main(x, y, a)
