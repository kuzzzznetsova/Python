import matplotlib.pyplot as plt
import numpy as np
import math
def extract_coef(A, B):
    d = B
    a = []
    b = []
    c = []
    a.append(0)
    b.append(A[0][0])
    c.append(A[0][1])
    for i in range(1, len(A)):
        a.append(A[i][i-1])
        b.append(A[i][i])
        if i != len(A)-1:
            c.append(A[i][i+1])
        else:
            c.append(0)
    return a, b, c, d

def method_progonkee(A, B):
    a, b, c, d = extract_coef(A, B)
    P = []
    Q = []
    for A, B, C in zip(a, b, c):
        if abs(B) < abs(A) + abs(C):
            raise Exception("Не выполнено условие для метода прогонки")
    P.append(-1*c[0]/b[0])
    Q.append(d[0]/b[0])

    for i in range(1, len(a)):
        p = -1*c[i]
        y = (b[i]+a[i]*P[i-1])
        p = p/y
        P.append(p)
        for j in range(len(P)):
            if P[j] == -0.0:
                P[j] = 0
        q = d[i] - a[i]*Q[i-1]
        q = q/y
        Q.append(q)
    X = []
    X.append(Q[len(Q)-1])

    for i in range(1, len(Q)):
        n = len(Q)-1
        x = P[n-i]*X[i-1] + Q[n-i]
        X.append(x)
    return X[::-1]

def cubic_spline(X, Y):
    n = len(X)
    h = [X[i] - X[i - 1] for i in range(1, n)]
    A = np.zeros((n - 2, n - 2))
    A[0][0] = 2 * (h[1] + h[2])
    A[0][1] = h[2]
    B = np.zeros(n - 2)
    B[0] = 3 * ((Y[2] - Y[1]) / h[2] - (Y[1] - Y[0]) / h[1])
    for i in range(3, n - 1):
        B[i - 2] = 3 * ((Y[i] - Y[i - 1]) / h[i] -
                        (Y[i - 1] - Y[i - 2]) / h[i - 1])
        A[i - 2][i - 3] = h[i - 1]
        A[i - 2][i - 2] = 2 * (h[i - 1] + h[i])
        A[i - 2][i - 1] = h[i]
    A[n - 3][n - 4] = h[n - 2]
    A[n - 3][n - 3] = (h[n - 3] + h[n - 2]) * 2
    B[n - 3] = 3 * ((Y[len(Y) - 1] - Y[len(Y) - 2]) /
                    h[n - 2] - (Y[len(Y) - 2] - Y[len(Y) - 3]) / h[n - 3])
    c = [0, 0] + method_progonkee(A, B)
    a = [f for f in Y[:len(Y) - 1]]
    h = [0] + h
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(1, n - 1):
        b[i] = (Y[i] - Y[i - 1]) / h[i] - (h[i] * (c[i + 1] + 2 * c[i])) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    b[n - 1] = (Y[n - 1] - Y[n - 2]) / h[n - 1] - \
        (2 * h[n - 1] * c[n - 1]) / 3
    d[n - 1] = -c[n - 1] / (3 * h[n - 1])
    a = [0] + a

    def spline_func(x):
        i = -1
        for k in range(1, n):
            if x >= X[k - 1] and x < X[k]:
                i = k
                break
        if (i == -1):
            return math.nan
        return a[i] + b[i]*(x-X[i - 1]) + c[i]*(x-X[i - 1])**2 + d[i]*(x-X[i - 1])**3
    return spline_func

def draw(X, Y, spline):
    x = np.arange(0, 7, 0.01)
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color="black", s=35)
    plt.plot(x, [float(spline(xk)) for xk in x], color="orange")
    plt.grid()
    plt.show()

def main():
    print('   Лабораторная работа №3')
    print('Кузнецова Дарина М8О-305Б-20')
    print("Кубический сплайн")
    X = [0.0, 1.7, 3.4, 5.1, 6.8]
    Y = [0, 1.3038, 1.8439, 2.2583, 2.6077]
    x0 = 3.0
    spline = cubic_spline(X, Y)
    print(f'Значение сплайна в точке {x0}: {spline(x0)}')
    draw(X, Y, spline)
main()

