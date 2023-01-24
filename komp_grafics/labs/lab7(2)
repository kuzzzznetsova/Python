#Кузнецова Дарина группа М8О-305Б-20
#Лабораторная работа №7
#Тема: Построение плоских полиномиальных кривых
#Кривая Безье 4-ой степени
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Entry

def clicked():
    #поля для ввода значений точек
    x1 = float(txt1.get())
    y1 = float(txt2.get())
    x2 = float(txt3.get())
    y2 = float(txt4.get())
    x3 = float(txt5.get())
    y3 = float(txt6.get())
    x4 = float(txt7.get())
    y4 = float(txt8.get())
    x5 = float(txt9.get())
    y5 = float(txt10.get())
#создаём окошко для вывода графика
    Title = "Кривая Безье 4-ой степени"
    plt.figure("Decart System")
    t = np.arange(0, 1, 0.001)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range (len(t)):
        x[i] = (1 - t[i])**4 * x1 + 4 * (1 - t[i])**3* t[i] * x2 + 4 * (1 - t[i])**2 * t[i]**2 * x3 + 4 * (1 - t[i]) * t[i]**3 * x4 + t[i]**4 * x5
        y[i] = (1 - t[i])**4 * y1 + 4 * (1 - t[i])**3* t[i] * y2 + 4 * (1 - t[i])**2 * t[i]**2 * y3 + 4 * (1 - t[i]) * t[i]**3 * y4 + t[i]**4 * y5
    plt.plot(x, y, 'b')
    #рисуем жирные точки
    plt.scatter(x1, y1, color='purple', s = 100, marker='o')
    plt.scatter(x2, y2, color='purple', s = 100, marker='o')
    plt.scatter(x3, y3, color='purple', s = 100, marker='o')
    plt.scatter(x4, y4, color='purple', s = 100, marker='o')
    plt.scatter(x5, y5, color='purple', s = 100, marker='o')
    plt.grid(True)
    plt.title(Title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

window = Tk() #создаём окошко для ввода данных
window.title("Кривая Безье 4-ой степени")
window.geometry("230x200") #устанавливаем размер экрана
lbl = Label(window, text = "Points(x,y):", font=("Arial Bold", 10))
lbl.place(x = 1, y = 1)
lb2 = Label(window, text = "x1:") #ввод значений первой координаты
lb2.place(x = 1, y = 25)
txt1 = Entry(window, width = 10)
txt1.place(x = 40, y = 25)
lb3 = Label(window, text = "     y1:")
lb3.place(x = 100, y = 25)
txt2 = Entry(window, width = 10)
txt2.place(x = 140, y = 25)

lb4 = Label(window, text = "x2:") #ввод значений второй координаты
lb4.place(x = 1, y = 50)
txt3 = Entry(window, width = 10)
txt3.place(x = 40, y = 50)
lb5 = Label(window, text = "     y2:")
lb5.place(x = 100, y = 50)
txt4 = Entry(window, width = 10)
txt4.place(x = 140, y = 50)

lb6 = Label(window, text = "x3:") #ввод значений третьей координаты
lb6.place(x = 1, y = 75)
txt5 = Entry(window, width = 10)
txt5.place(x = 40, y = 75)
lb7 = Label(window, text = "     y3:")
lb7.place(x = 100, y = 75)
txt6 = Entry(window, width = 10)
txt6.place(x = 140, y = 75)

lb8 = Label(window, text = "x4:") #ввод значений четвертой координаты
lb8.place(x = 1, y = 100)
txt7 = Entry(window, width = 10)
txt7.place(x = 40, y = 100)
lb9 = Label(window, text = "     y4:")
lb9.place(x = 100, y = 100)
txt8 = Entry(window, width = 10)
txt8.place(x = 140, y = 100)

lb10 = Label(window, text = "x5:") #ввод значений пятой координаты
lb10.place(x = 1, y = 125)
txt9 = Entry(window, width = 10)
txt9.place(x = 40, y = 125)
lb11 = Label(window, text = "     y5:")
lb11.place(x = 100, y = 125)
txt10 = Entry(window, width = 10)
txt10.place(x = 140, y = 125)
#Кнопка "Построить"
btn = Button(window, text = "Построить", bg = "steelblue", fg = "white", command = clicked)
btn.place(x = 80, y = 160)
window.mainloop()
