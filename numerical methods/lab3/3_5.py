from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

def function(x):
    return 1 / (3 * x**2 + 4 * x + 2)

def rectangles(h, x_0, x_k):
    area = 0
    while x_0 < x_k:
        x_next = x_0 + h
        area += function(x_0 / 2 + x_next / 2) * h
        x_0 += h
    return area

def Trapez_method(h, X0, Xk):
    n = int((Xk - X0) / h) + 1
    x = np.linspace(X0, Xk, n)
    result = 0
    for i in range(1, n):
        result += h / 2 * (function(x[i]) + function(x[i - 1]))
    return result

def parabolas(h, x_0, x_k):
    area = function(x_0) * h / 3
    iteration = 1
    while x_0 < x_k:
        x_next = x_0 + h
        if iteration % 2 != 0 and x_next < x_k:
            area += 4 * function(x_next) * h / 3
        if iteration % 2 == 0 and x_next < x_k:
            area += 2 * function(x_next) * h / 3
        if x_next == x_k and (iteration - 1) % 2 != 0:
            area += function(x_next) * h / 3
        iteration += 1
        x_0 += h
    return area

def adjusment(y1, y2, h1, h2, p):
    if h1 > h2:
        h1, h2 = h2, h1
        y1, y2 = y2, y1
    k = h2 / h1
    return y1 + (y1 - y2) / (k ** p - 1)

def main():
    print('   Лабораторная работа №3')
    print('Кузнецова Дарина М8О-305Б-20')
    x_0 = -2
    x_k = 2
    h_1 = 1
    h_2 = 0.5
    rects_h1 = rectangles(h_1, x_0, x_k)
    rects_h2 = rectangles(h_2, x_0, x_k)
    rects_adjusted = adjusment(rects_h1, rects_h2, h_1, h_2, 1)

    traps_h1 = Trapez_method(h_1, x_0, x_k)
    traps_h2 = Trapez_method(h_2, x_0, x_k)
    traps_adjusted = adjusment(traps_h1, traps_h2, h_1, h_2, 1)

    parabs_h1 = parabolas(h_1, x_0, x_k)
    parabs_h2 = parabolas(h_2, x_0, x_k)
    parabs_adjusted = adjusment(parabs_h1, parabs_h2, h_1, h_2, 1)

    print("Метод прямоугольников:")
    print(f"h = {h_1}:\t{rects_h1}")
    print(f"h = {h_2}:\t{rects_h2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона:\t{rects_adjusted}")
    print("")

    print("Погрешности:")
    print(f"{abs(rects_adjusted - rects_h1)}")
    print(f"{abs(rects_adjusted - rects_h2)}")
    print("")
    print("-------------------")
    print("")

    print("Метод трапеций:")
    print(f"h = {h_1}:\t{traps_h1}")
    print(f"h = {h_2}:\t{traps_h2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона:\t{traps_adjusted}")
    print("")

    print("Погрешности:")
    print(f"{abs(traps_adjusted - traps_h1)}")
    print(f"{abs(traps_adjusted - traps_h2)}")
    print("")
    print("-------------------")
    print("")

    print("Метод Симпсона:")
    print(f"h = {h_1}:\t{parabs_h1}")
    print(f"h = {h_2}:\t{parabs_h2}")
    print(f"Уточнение методом Рунге-Ромберга-Ричардсона:\t{parabs_adjusted}")
    print("")

    print("Погрешности:")
    print(f"{abs(parabs_adjusted - parabs_h1)}")
    print(f"{abs(parabs_adjusted - parabs_h2)}")
    print("")
    print("-------------------")
    print("")

    print("Проверка метода P-P-P\nПoиcк тoчнoгo значения интеграла и погрешность:")

    calculated = quad(function, 0, 4)
    print(calculated)
main()
