import numpy as np
import matplotlib.pyplot as plt


def given_function(x):
    return np.cos(x)


def f_ij(dots):
    if len(dots) == 2:
        return (
            given_function(dots[0]) - given_function(dots[1])
        ) / (
            dots[0] - dots[1]
        )
    else:
        return (
            f_ij(dots[:len(dots) - 1]) - f_ij(dots[1:])
        ) / (
            dots[0] - dots[len(dots) - 1]
        )


def product_from_numbers(x, arr):
    result = 1

    for number in arr:
        result *= x - number

    return result


def product_from_callable(x, fn_arr):
    result = 1

    for fn in fn_arr:
        if callable(fn):
            result *= fn(x)
        else:
            result *= fn

    return result


def with_product_numbers(arr):
    def hof(x):
        return product_from_numbers(x, arr)

    return hof


def with_product_callable(arr):
    def hof(x):
        return product_from_callable(x, arr)

    return hof


def with_plus(functions):
    def hof(x):
        result = 0

        for fn in functions:
            if callable(fn):
                result += fn(x)
            else:
                result += fn

        return result

    return hof


def build_lagrange(dots):
    def l_polynom(_x):
        return 0

    for x in dots:
        iterable = list(dots)
        iterable.remove(x)
        f_koeff = given_function(x)

        multi_omega_fn = with_product_numbers(iterable)
        omega = multi_omega_fn(x)

        l_polynom = with_plus([
            l_polynom,
            with_product_callable([f_koeff/omega, multi_omega_fn])
        ])

    return l_polynom


def build_newton(dots):
    p_x = given_function(dots[0])

    iterable = [dots[0]]

    for x in dots[1:]:
        x_little_polynom = with_product_numbers(iterable)

        iterable.append(x)

        f_koeff = f_ij(iterable)

        p_x = with_plus([
            p_x,
            with_product_callable([f_koeff, x_little_polynom])
        ])

    return p_x


def draw(x, y, dots, y_i, polynom_lagrange, polynom_newton):

    plt.figure(figsize=(10, 6))

    all_dots = [0, np.pi/6, np.pi/3, np.pi/2]

    plt.scatter(dots, given_function(dots), color="black", s=35)
    plt.plot(x, y, linewidth=1, color="green")
    plt.plot(x, [polynom_lagrange(xi) for xi in x],
             label='Лагранж L(x)', linewidth=1, color="red")
    plt.plot(x, [float(polynom_newton(xi)) for xi in x],
             label='Ньютон P(x)', linewidth=1, color="orange")
    plt.legend()
    plt.grid()
    plt.show()


def solve(dots):
    x = np.linspace(-1, 2, 100)
    y = given_function(x)
    y_i = given_function(dots)
    polynom_lagrange = build_lagrange(dots)
    polynom_newton = build_newton(dots)

    X_asterisk = np.pi / 4

    print("\nЛагранж L(X*) = ",
          np.around(polynom_lagrange(X_asterisk), 3))
    polynom_newton = polynom_lagrange
    print("y(X*) = ",
          np.around(given_function(X_asterisk), 2))
    print("Абсолютная погрешность интерполяции составляет: ",
          (np.around(polynom_lagrange(X_asterisk) - given_function(X_asterisk), 3)))
    print("")
    print("Ньютон P(X*) = ",
          polynom_newton(X_asterisk))
    print("y(X*) = ",
          np.around(given_function(X_asterisk), 2))
    print("Абсолютная погрешность интерполяции составляет: ",
          (np.around(polynom_newton(X_asterisk) - given_function(X_asterisk), 3)))

    # многочлен лагранжа и ньютона получаются одинаковыми, на графике сливаются
    draw(x, y, dots, y_i, polynom_lagrange, polynom_newton)


print("\n")

solve(np.array([0, np.pi/6, np.pi/3]))
# solve(np.array([np.pi/6, np.pi/3, np.pi/2]))
# solve(np.array([0, np.pi/6, np.pi/3, np.pi/2]))

solve(np.array([0, np.pi/6, 5*np.pi/12, np.pi/2]))

print("")
