# Лабораторная работа №3
# Тема: Основы построения фотореалистичных изображений
# Вариант № 17 : Прямой цилиндр, основание – сектор гиперболы
# Кузнецова Дарина
# группа: М8О-305Б-20
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from tkinter import *

def get_vertices(L, B, I):
    P = np.empty((6, 3), dtype=float)
    Ind = np.array([3, 4, 5])

    P[Ind, 2] = I
    P[3, 0] = L+1
    P[3, 1] = 0
    P[4, 0] = 1
    P[4, 1] = 0 + 0.000000000002
    P[5, 0] = 1
    P[5, 1] = L + 0.0000001

    Ind = Ind - 3
    P[Ind, 2] = 0
    P[0, 0] = L+1 + 0.000000000001
    P[0, 1] = 0 + 0.00000001
    P[1, 0] = 1+ 0.00001
    P[1, 1] = 0
    P[2, 0] = 1 + 0.000001
    P[2, 1] = L
    return P

def draw(P):
    Title = "Фигура до аппроксимации"
    l_x = -7
    r_x = 7
    l_y = -7
    r_y = 7
    l_z = -0.5
    r_z = 7
    fig = plt.figure('Лабораторная работа №3')
    plt.title(Title)

    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_y, r_y))
    ax.set_zlim((l_z, r_z))
    plt.axis('off')

    V = [[P[1], P[4], P[3], P[0]], [P[0], P[3], P[5], P[2]], [P[2], P[5], P[4], P[1]], [P[3], P[4], P[5]], [P[0], P[1], P[2]]]
    ax.add_collection3d(Poly3DCollection(V, facecolors='black', linewidths=2, edgecolors='gray', alpha=0.99, cmap='inferno'))
    plt.show()

    # ax.plot_trisurf([P[0, 0], P[3, 0], P[4, 0],P[1, 0]], [P[0, 1], P[3, 1], P[4, 1],P[1, 1]],
    #                 [P[0, 2], P[3, 2], P[4, 2], P[1, 2]], color='black', linewidth=0, antialiased=False)
    # ax.plot_trisurf([P[0, 0], P[3, 0], P[5, 0], P[2, 0]], [P[0, 1], P[3, 1], P[5, 1], P[2, 1]],
    #                 [P[0, 2], P[3, 2], P[5, 2], P[2, 2]], color='black', linewidth=0, antialiased=False)
    # ax.plot_trisurf([P[2, 0], P[5, 0], P[4, 0], P[1, 0]], [P[2, 1], P[5, 1], P[4, 1], P[1, 1]],
    #                 [P[2, 2], P[5, 2], P[4, 2], P[1, 2]], color='black', linewidth=0, antialiased=False)
    # ax.plot_trisurf([P[3, 0], P[4, 0], P[5, 0]], [P[3, 1], P[4, 1], P[5, 1]],
    #                 [P[3, 2], P[4, 2], P[5, 2]], color='gray', linewidth=0, antialiased=False)
    # ax.plot_trisurf([P[0, 0], P[1, 0], P[2, 0]], [P[0, 1], P[1, 1], P[2, 1]],
    #                 [P[0, 2], P[1, 2], P[2, 2]], color='gray', linewidth=0, antialiased=False)
    # plt.show()
    # return None

def draw2(a):
    global L
    l_x = -7
    r_x = 7
    l_y = -7
    r_y = 7
    l_z = -0.5
    r_z = 7
    Title = "Фигура после аппроксимации (при a = " + (str(a)) + ")"
    fig = plt.figure('Лабораторная работа №3')
    plt.title(Title)

    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlim((l_x, r_x))
    ax.set_ylim((l_y, r_y))
    ax.set_zlim((l_z, r_z))
    plt.axis('off')

    tmp1 = list()
    tmp2 = list()

    if a == 1:
        draw(points)
        return

    del_xy = L * 2 * 0.95/a
    x_0 = points[2][0]
    y_0 = points[2][1]
    for i in range(a//2-1):
        y_0 -= del_xy
        x_0 = 1/y_0
        tmp1.append([x_0, y_0, points[0][2]])
        tmp2.append([x_0+0.00000001, y_0+0.000000001, points[3][2]])

    tmp1_0 = list()
    tmp2_0 = list()
    x_0 = points[0][0]
    y_0 = points[0][1]
    for i in range(a//2):
        x_0 -= del_xy
        y_0 = 1/x_0
        tmp1_0.append([x_0, y_0, points[0][2]])
        tmp2_0.append([x_0+0.000001, y_0+0.0000001, points[3][2]])
    tmp1_0.reverse()
    tmp2_0.reverse()
    tmp1 = tmp1 + tmp1_0
    tmp2 = tmp2 + tmp2_0
    # tmp1.append(points[0])
    # tmp1.append(points[2])
    # tmp2.append(points[3])
    # tmp2.append(points[5])

    V = list()
    for i in range(len(tmp1)):
        if i==len(tmp1)-1:
            V.append([tmp1[i], tmp1[0], tmp2[0], tmp2[i]])
        else:
            V.append([tmp1[i], tmp1[i+1], tmp2[i+1], tmp2[i]])

    tmp = list()
    for j in range(len(tmp1)):
        tmp.append(tmp1[j])
    V.append(tmp)

    tmp.clear()
    for j in range(len(tmp2)):
        tmp.append(tmp2[j])
    V.append(tmp)

    # for i in range(len(V)-2):
    #     ax.plot_trisurf([V[i][0][0], V[i][1][0], V[i][2][0], V[i][3][0]], [V[i][0][1], V[i][1][1], V[i][2][1], V[i][3][1]],
    #                     [V[i][0][2], V[i][1][2], V[i][2][2], V[i][3][2]], color='black', linewidth=0, antialiased=False)
    ax.add_collection3d(Poly3DCollection(V, facecolors='black', linewidths=0.5, edgecolors='gray', alpha=0.99, cmap = 'jet'))
    plt.show()
    close_window_draw2(a)

#функция для закрытия окошка для ввода
def close_window():
    a = txt_1.get()
    window.destroy()
    draw2(int(a))
    file = open("File.txt", "w")
    file.write(str(a))

def close_window_0():
    global points, L, B, I
    L = int(txt_1.get())
    B = L * np.sqrt(2)
    I = int(txt_2.get())
    points = get_vertices(L, B, I)
    window_0.destroy()
    draw(points)

def close_window_1():
    my_file = open("File.txt")
    a = my_file.readline()
    draw2(int(a))


def close_window_draw2(a):
    file = open("File.txt", "w")
    file.write(str(a))
    window_4 = Tk()
    window_4.title("Лабораторная работа №3")
    window_4.geometry('400x250')
    lbl = Label(window_4, text="Значение аппроксимации (а) добавлено в файл 'File.txt'", font=("Arial Bold", 10))
    lbl.grid(column=0, row=5)
    btn = Button(window_4, text="Открытие файла", command = window_4.quit)
    btn.grid(column=0, row=20)
    window_4.mainloop()
    open_my_file(a)

def open_my_file(a):
    my_file = open("File.txt")
    # a1 = my_file.readline()
    window_5 = Tk()
    window_5.title("Лабораторная работа №3")
    lbl = Label(window_5, text="В файле 'File.txt' находится значение а =" + (str(a)) ,font=("Arial Bold", 10))
    lbl.grid(column=0, row=5)
    btn = Button(window_5, text="Готово", command = window_5.quit)
    btn.grid(column=0, row=20)
    window_5.mainloop()

points = list()

window_0 = Tk()
window_0.title("Лабораторная работа №3")
window_0.geometry('400x250')
lbl = Label(window_0, text="Введите параметры фигуры (L-сторона, I-высота)", font=("Arial Bold", 10))
lbl.grid(column=0, row=0)

txt_1 = Entry(window_0, width=20)
txt_1.grid(column=0, row=20)
txt_1.focus()

txt_2 = Entry(window_0, width=20)
txt_2.grid(column=0, row=40)

btn_0 = Button(window_0, text="Построить!", command=close_window_0)
btn_0.grid(column=0, row=60)

window_0.mainloop()


window = Tk()
window.title("Лабораторная работа №3")
window.geometry('400x250')

lbl1 = Label(window, text="Введите значение аппроксимации", font=("Arial Bold", 10))
lbl1.grid(column=0, row=5)
txt_1 = Entry(window, width=20)
txt_1.grid(column=0, row=10)
txt_1.focus()

lbl2 = Label(window, text="Введение значение освещенности", font=("Arial Bold", 10))
lbl2.grid(column=0, row=15)
txt_2 = Entry(window, width=20)
txt_2.grid(column=0, row=20)

lbl3 = Label(window, text="Введите значение отражающей способности", font=("Arial Bold", 10))
lbl3.grid(column=0, row=25)
txt_3 = Entry(window, width=20)
txt_3.grid(column=0, row=30)

btn = Button(window, text="Построить!", command=close_window)
btn.grid(column=0, row=80)

btn1 = Button(window, text="Построить со старым значением аппроксимации!", command=close_window_1)
btn1.grid(column=0, row=140)

window.mainloop()
