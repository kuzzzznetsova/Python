# Лабораторная работа №2
# Тема: Каркасная визуализация выпуклого многогранника. Удаление невидимых линий.
# Вариант № 20 : 4 – гранная прямая правильная усеченная пирамида
# Кузнецова Дарина
# группа: М8О-305Б-20

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def get_vertices(L, B, I, alpha, beta):
    P = np.empty((8, 3), dtype=float)
    Ind = np.array([4, 5, 6, 7])

    P[Ind, 0] = L / 2 + I * np.tan(np.pi * alpha / 180)
    P[Ind, 1] = B / 2 + I * np.tan(np.pi * beta / 180)
    P[Ind, 2] = I
    P[5, 0] = - P[5, 0]
    P[7, 1] = - P[7, 1]
    P[6, [0, 1]] = - P[6, [0, 1]]

    Ind = Ind - 4
    P[Ind, 0] = L / 2
    P[Ind, 1] = B / 2
    P[Ind, 2] = 0
    P[1, 0] = - P[1, 0]
    P[3, 1] = - P[3, 1]
    P[2, [0, 1]] = - P[2, [0, 1]]
    return P

def draw(P):
    Title = "4 – гранная прямая правильная усеченная пирамида"
    l_x = -7
    r_x = 7
    l_y = -7
    r_y = 7
    l_z = -0.5
    r_z = 7
    fig = plt.figure('Лабораторная работа №2 Кузнецова М8О-305Б-20')
    plt.title(Title)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_y, r_y))
    ax.set_zlim((l_z, r_z))
    plt.xlabel("ось x")
    plt.ylabel("ось y")

    ax.plot_trisurf([P[0, 0], P[1, 0], P[5, 0], P[4, 0]], [P[0, 1], P[1, 1], P[5, 1], P[4, 1]],
                    [P[0, 2], P[1, 2], P[5, 2], P[4, 2]], color='#00BFFF', linewidth=0, antialiased=False)
    ax.plot_trisurf([P[0, 0], P[4, 0], P[7, 0], P[3, 0]], [P[0, 1], P[4, 1], P[7, 1], P[3, 1]],
                    [P[0, 2], P[4, 2], P[7, 2], P[3, 2]], color='#1E90FF', linewidth=0, antialiased=False)
    ax.plot_trisurf([P[3, 0], P[7, 0], P[6, 0], P[2, 0]], [P[3, 1], P[7, 1], P[6, 1], P[2, 1]],
                    [P[3, 2], P[7, 2], P[6, 2], P[2, 2]], color='#ADD8E6', linewidth=0, antialiased=False)
    ax.plot_trisurf([P[4, 0], P[5, 0], P[6, 0], P[7, 0]], [P[4, 1], P[5, 1], P[7, 1], P[6, 1]],
                    [P[4, 2], P[5, 2], P[7, 2], P[6, 2]], color='#87CEFA', linewidth=0, antialiased=False)
    ax.plot_trisurf([P[1, 0], P[5, 0], P[6, 0], P[2, 0]], [P[1, 1], P[5, 1], P[6, 1], P[2, 1]],
                    [P[1, 2], P[5, 2], P[6, 2], P[2, 2]], color='#4682B4', linewidth=0, antialiased=False)
    ax.plot_trisurf([P[0, 0], P[1, 0], P[2, 0], P[3, 0]], [P[0, 1], P[1, 1], P[2, 1], P[3, 1]],
                    [P[0, 2], P[1, 2], P[2, 2], P[3, 2]], color='#00BFFF', linewidth=0, antialiased=False)
    # ax.plot_trisurf(P[0:4, 0], P[0:4, 1], P[0:4, 2], color='#4682B4', linewidth=0, antialiased=False)
    plt.show()
    return None

def draw2(P):
    Title = "каркасное представление"
    l_x = -7
    r_x = 7
    l_y = -7
    r_y = 7
    l_z = -0.5
    r_z = 7
    fig2 = plt.figure('Лабораторная работа №2 Кузнецова М8О-305Б-20')
    plt.title(Title)

    ax = fig2.add_subplot(111,projection='3d')
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_y, r_y))
    ax.set_zlim((l_z, r_z))
    plt.axis('off')

    Ind = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    V = [[P[1], P[5], P[4], P[0]],[P[0], P[4], P[7], P[3]],[P[3], P[7], P[6], P[2]],[P[1], P[2], P[6], P[5]], [P[1],P[2], P[3],P[0]], [P[4], P[5], P[6], P[7]]]
    ax.add_collection3d(Poly3DCollection(V, facecolors='w', linewidths=1, edgecolors='black', alpha=0.99))
    plt.show()
    # for i in Ind:
    #     ax.plot(P[i, 0], P[i, 1], P[i, 2], 'b-')
    # Ind = np.array([[4, 5], [5, 6], [6, 7], [7, 4]])
    # for i in Ind:
    #     ax.plot(P[i, 0], P[i, 1], P[i, 2], 'b-')
    # Ind = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
    # for i in Ind:
    #     ax.plot(P[i, 0], P[i, 1], P[i, 2], 'b-')
    # ax.plot(P[:, 0], P[:, 1], P[:, 2], 'bo')
    # plt.show()
    # return None

def plot_XOY(P):
    Title = "Проекция OXY"
    fig3, ax = plt.subplots()
    plt.title(Title)
    l_x = -7
    r_x = 7
    l_y = -7
    r_y = 7
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_y, r_y))
    plt.xlabel("ось x")
    plt.ylabel("ось y")

    ax.plot([0], [0])
    ax.add_patch(Rectangle((P[0, 0], P[0, 1]), P[1, 0] - P[0, 0], P[3, 1] - P[0, 1]))
    plt.show()
    return None

def plot_XOZ(P):
    Title = "Проекция OXZ"
    fig4, ax = plt.subplots()
    plt.title(Title)
    l_x = -7
    r_x = 7
    l_z = -0.5
    r_z = 7
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_z, r_z))
    plt.xlabel("ось x")
    plt.ylabel("ось z")

    points = [[P[2, 0], P[2, 2]], [P[3, 0], P[3, 2]], [P[7, 0], P[7, 2]], [P[6, 0], P[6, 2]], [P[2, 0], P[2, 2]]]
    polygon = plt.Polygon(points)
    ax.add_patch(polygon)
    plt.show()
    return None


def plot_YOZ(P):
    Title = "Проекция OYZ"
    fig5, ax = plt.subplots()
    plt.title(Title)
    l_y = -7
    r_y = 7
    l_z = -0.5
    r_z = 7
    ax.set_xlim((l_y, r_y))
    ax.set_ylim((l_z, r_z))
    plt.xlabel("ось y")
    plt.ylabel("ось z")

    points = [[P[1, 1], P[1, 2]], [P[2, 1], P[2, 2]], [P[6, 1], P[6, 2]], [P[5, 1], P[5, 2]], [P[1, 1], P[1, 2]]]
    polygon = plt.Polygon(points)
    ax.add_patch(polygon)
    plt.show()
    return None


def plot_IZO(P):
    Title = "Изометрия"
    fig5, ax = plt.subplots()
    plt.title(Title)
    l_y = -5
    r_y = 5
    l_z = -5
    r_z = 5
    ax.set_xlim((l_y, r_y))
    ax.set_ylim((l_z, r_z))

    x = np.array([3, -2])
    y = np.array([-3, -2])
    z = np.array([-0.08, 4])

    arrowprops = {
        'arrowstyle': '<|-',
    }
    plt.annotate(r'x',
                 xy=(0, 0),
                 xytext=x,
                 arrowprops=arrowprops)
    plt.annotate(r'y',
                 xy=(0, 0),
                 xytext=y,
                 arrowprops=arrowprops)
    plt.annotate(r'z',
                 xy=(0, 0),
                 xytext=z,
                 arrowprops=arrowprops)
    plt.grid()

    alpha = -35
    beta = -45

    C = np.empty((8, 3), dtype=float)
    A1 = [[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]
    A2 = [[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]]
    mul = np.dot(A1, A2)
    for j in range(8):
        C[j] = np.dot(mul, P[j])
    # points = [[C[0, 0], C[0, 1]], [C[1, 0], C[1, 1]], [C[5, 0], C[5, 1]], [C[6, 0], C[6, 1]], [C[7, 0], C[7, 1]],[C[3, 0], C[3, 1]],[C[0, 0], C[0, 1]]]
    # polygon = plt.Polygon(points)
    # ax.add_patch(polygon)
    # plt.show()
    Ind = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    for i in Ind:
        ax.plot(C[i, 1], C[i, 0], 'b-')
    Ind = np.array([[4, 5], [5, 6], [6, 7], [7, 4]])
    for i in Ind:
        ax.plot(C[i, 1], C[i, 0], 'b-')
    Ind = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
    for i in Ind:
        ax.plot(C[i, 1], C[i, 0], 'b-')

    plt.show()
    return None

L = 6
B = 6
I = 4
alpha = -20
beta = -20

ful = get_vertices(L, B, I, alpha, beta)
draw(ful)

ful2 = get_vertices(L, B, I, alpha, beta)
draw2(ful2)

ful3 = get_vertices(L, B, I, alpha, beta)
plot_XOY(ful3)

ful4 = get_vertices(L, B, I, alpha, beta)
plot_XOZ(ful4)

ful5 = get_vertices(L, B, I, alpha, beta)
plot_YOZ(ful4)

ful6 = get_vertices(L, B, I, alpha, beta)
plot_IZO(ful6)
