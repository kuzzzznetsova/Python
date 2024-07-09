# Лабораторная работа №1
# Тема: Построение изображений 2D- кривых
# Вариант № 21 : ρ = a*cos(7φ)
# Кузнецова Дарина
# группа: М8О-305Б-20
import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.widgets import Slider
from tkinter import *

def func(a):
    Title = "p =" + str(a) + "* cos(7*phi)"
    phi = np.arange(0,2*np.pi, np.pi/180)
    ro = a * np.cos(7*phi)
    #Полярная система координат
    plt.figure("Polar System")
    plt.polar(phi, ro)
    plt.grid(True)
    plt.title(Title)

    #Декартова система координат
    plt.figure("Decart System")
    x = ro * np.cos(phi)
    y = ro * np.sin(phi)
    plt.plot(x, y, 'r')
    plt.grid(True)
    plt.title(Title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

#функция для закрытия окошка для ввода
def close_window():
    a = txt.get()
    window.destroy()
    func(float(a))

#создание окна для ввода данных
window = Tk()
window.title("окошко для ввода")
window.geometry('400x250')

lbl = Label(window, text="ρ = a*cos(7φ)", font=("Arial Bold", 15))
lbl.grid(column=0, row=0)

lbl1 = Label(window, text="введите значения", font=("Arial Bold", 15), fg="indigo")
lbl1.grid(column=3, row=3)

lbl2 = Label(window, text="a =", font=("Arial Bold", 10))
lbl2.grid(column=2, row=5)

txt = Entry(window, width=20)
txt.grid(column=3, row=5)
txt.focus()

btn = Button(window, text="Построить!", bg="thistle", fg="indigo", command=close_window)
btn.grid(column=3, row=100)

window.mainloop()
