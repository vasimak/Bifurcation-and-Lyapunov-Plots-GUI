# PlotGui

#from asyncio.streams import _ClientConnectedCallback
import re
from math import log
from cmath import inf
from cmath import sqrt
from cmath import cos
from cmath import acos
import PySimpleGUI as sg
from functools import reduce
import numpy as np
import math
import csv
import time
from functools import partial
from numba import jit
import gc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.figure as figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import multiprocessing 
from multiprocessing import Pool
matplotlib.use('TkAgg')

sg.theme('Black')
haserror = False
X = []
Y = []
X1 = []
Y1 = []
fig = figure.Figure()
ax = fig.add_subplot(111)
DPI = fig.get_dpi()
fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
plt.style.use('dark_background')

# ------------------------------- This is to include a matplotlib figure in a Tkinter canvas
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle((min(x1, x2), min(y1, y2)),
                         np.abs(x1-x2), np.abs(y1-y2))
    print(rect)
    ax.add_patch(rect)
    fig.canvas.draw()


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


@jit(nopython=True, parallel=True)
def bif(q0, x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    q = q0
    for i in range(1, N):
        x[i] = r * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q
    return (x[-130:])


@jit(nopython=True, parallel=True)
def le(q0, x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    q = q0
    for i in range(1, N):
        x = r * (1 + x) * (1 + x) * (2 - x) + q
      
        lyapunov += np.log(np.abs(-3*r*(x**2-1)))
        l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def log_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return (x[-130:])


@jit(nopython=True, parallel=True)
def log_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = x = r * x * (1 - x)
      
        lyapunov += np.log(np.abs(r - 2*r*x))
        l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def cheb_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = math.cos(r*math.acos(x[i-1]))
    return (x[-130:])


@jit(parallel=True)
def cheb_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = math.cos(r*math.acos(x))
      
        if math.sqrt(1-x**2)==0:
            r*math.sin(r*math.acos(x)) ==0
        else:
            lyapunov += np.log(np.abs((r*math.sin(r*math.acos(x))) /
                           (math.sqrt(1-x**2))))
        l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def sine_sinh_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = r*math.sin(math.pi*math.sinh(math.pi*math.sin(math.pi*x[i-1])))
    return (x[-130:])


@jit(nopython=True)
def sine_sinh_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = r*math.sin(math.pi*math.sinh(math.pi*math.sin(math.pi*x)))
      
        lyapunov += np.log(np.abs(math.pi**3*r*math.cos(math.pi*x)*math.cosh(
            math.pi*math.sin(math.pi*x))*math.cos(math.pi*math.sinh(math.sin(math.pi*x)))))
        l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def renyi_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = np.mod(r*x[i-1], 1)
    return (x[-130:])


@jit(nopython=True)
def renyi_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = np.mod(r*x, 1)
      
        lyapunov += np.log(np.abs(np.mod(r, 1)))
        l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def sine_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = r*math.sin(math.pi*x[i-1])
    return (x[-130:])


@jit(nopython=True)
def sine_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = r*math.sin(math.pi*x)
      
        lyapunov += np.log(np.abs(math.pi*r*math.cos(math.pi*x)))
        l1 = lyapunov/N
    return (l1)


jit(nopython=True, parallel=True)
def cubic_logistic_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = r*x[i-1]*(1-x[i-1])*(2+x[i-1])
    return (x[-130:])


@jit(nopython=True)
def cubic_logistic_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = r*x*(1-x)*(2+x)
      
        lyapunov += np.log(np.abs(-r*(3*x**2+2*x-2)))
        l1 = lyapunov/N
    return (l1)


jit(nopython=True, parallel=True)
def cubic_bif(x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    for i in range(1, N):
        x[i] = r*x[i-1]*(1-x[i-1]**2)
    return (x[-130:])


@jit(nopython=True)
def cubic_le(x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    for i in range(1, N):
        x = r*x*(1-x**2)
      
        lyapunov += np.log(np.abs(r-3*r*x**2))
        l1 = lyapunov/N
    return (l1)


jit(nopython=True, parallel=True)
def extracheb_bif(q0, x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    q = q0
    for i in range(1, N):
        x[i] = math.cos(r**q * math.acos(q*x[i - 1]))
     
    return (x[-130:])



@jit(nopython=True)
def extracheb_le(q0, x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    q = q0
    x = x0
    for i in range(1, N):
        x = math.cos(r**q * math.acos(q*x))
        if math.sqrt(1-(q**2*x**2))==0:
            q*r**q*math.sin(r**q * math.acos(q*x))==0
        else:
          
            lyapunov += np.log(np.abs((q*r**q*math.sin(r**q *
                            math.acos(q*x)))/(math.sqrt(1-(q**2*x**2)))))
            l1 = lyapunov/N
    return (l1)


@jit(nopython=True, parallel=True)
def extrasine_sinh_bif(q0, x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    q = q0
    for i in range(1, N):
        x[i] = r * math.sin(r * math.sinh(q * math.sin(2 * x[i - 1])))
    return (x[-130:])


@jit(nopython=True)
def extrasine_sinh_le(q0, x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    q = q0
    for i in range(1, N):
        x = r * math.sin(r * math.sinh(q * math.sin(2 * x)))
      
        lyapunov += np.log(np.abs(2*q*r**2*math.cos(2*x)
                           * math.cosh(q*math.sin(2*x))))
        l1 = lyapunov/N
    return (l1)


def extralogistic_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()
    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):

                if i <= 4:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[2] == values[3] or values[2] >= values[3]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[4]) < 0 or float(values[4]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(bif, q0, x0)
        le1 = partial(le, q0, x0)
        
        if event == 'Bifurcation Plot':
            start_time = time.time()       
            X = []
            Y = []
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue
            
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)
                  

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            print("--- %s seconds ---" % (time.time() - start_time))
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.axhline(0)
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue

    window.close()


def logistic_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(300 * 2, 300)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                    o = all((bool(re.fullmatch(
                        "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(log_bif, x0)
        le1 = partial(log_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def chebysev_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")

                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(cheb_bif, x0)
        le1 = partial(cheb_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def sine_sinh_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")

                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(sine_sinh_bif, x0)
        le1 = partial(sine_sinh_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def renyi_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")

                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(renyi_bif, x0)
        le1 = partial(renyi_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def sine_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")

                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(sine_bif, x0)
        le1 = partial(sine_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def cubic_logistic_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")

                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(cubic_logistic_bif, x0)
        le1 = partial(cubic_logistic_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def cubic_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 3:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[1] == values[2] or values[1] >= values[2]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[3]) < 0 or float(values[3]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[1]), float(values[2]), float(values[3]))
        x0 = float(values[0])
        bif1 = partial(cubic_bif, x0)
        le1 = partial(cubic_le, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def extracheb_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()
    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):

                if i <= 4:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[2] == values[3] or values[2] >= values[3]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[4]) < 0 or float(values[4]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(extracheb_bif, q0, x0)
        le1 = partial(extracheb_le, q0, x0)



        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue
        
        
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
          
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            try:
                for i, ch in enumerate(map(le1, r)):
                    X1.append(r[i])
                    Y1.append(ch)
            except ZeroDivisionError:
                sg.popup("again")
                continue
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def extrasine_sinh_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Step',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Button('Exit', size=(20, 1))],
        [sg.T('Here you can control the Plot:')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],

    ]
    window = sg.Window("Plots", layout,
                       resizable=True, finalize=True, grab_anywhere=True)
    window.Maximize()
    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):

                if i <= 4:
                    values_new[key] = values[key]

                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    haserror = True
                    break
                else:
                    haserror = False

            if haserror == True:
                continue
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            if values[2] == values[3] or values[2] >= values[3]:
                sg.popup("'End of r' can't be larger than 'Initial r' ")
                continue
            elif float(values[4]) < 0 or float(values[4]) > 1:
                sg.popup("Only numbers between 0 and 1")
                continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(extrasine_sinh_bif, q0, x0)
        le1 = partial(extrasine_sinh_le, q0, x0)

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".w", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
        
            try:
                for i, ch in enumerate(map(le1, r)):
                    X1.append(r[i])
                    Y1.append(ch)

            except ZeroDivisionError:
                    sg.popup("Try again with different numbers. It raises math error")
                    continue

            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X1, Y1, ".w", alpha=1, ms=1.2)
            ax.axhline(0)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            try:
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
            except ValueError:
                sg.popup("Try again with different numbers. It raises math error")
                continue

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".w", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            ax.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

            window.refresh()
            continue

    window.close()


def main():
    layout = [[sg.Text('READ THE HELP FIRST and then choose the Map you want to run:')],
              [sg.Text('1.'), sg.Button('Logistic Map', key="open3"),sg.Text('x(i) = r * x * (1 - x(i-1))')],
              [sg.Text('2.'), sg.Button('Chebyshev Map', key="open1"),sg.Text('x(i) = cos( r ^ q * arccos(q * x(i-1)))')],
              [sg.Text('3.'), sg.Button('Sine-Sinh Map', key="open2"),sg.Text('x(i) = r * sin(r * sinh(q * sin(2 * x(i - 1))))')],
              [sg.Text('4.'), sg.Button('Sine Map', key="open4"),sg.Text('x(i) = r * sin( π * x(i-1))')],
              [sg.Text('5.'), sg.Button('Renyi Map', key="open5"),sg.Text('x(i) = mod( r * x(i - 1), 1)')],
              [sg.Text('6.'), sg.Button('Cubic Logistic Map', key="open6"),sg.Text('x(i) = r * x(i - 1) * (1 - x(i - 1)) * (2 + x(i - 1))')],
              [sg.Text('7.'), sg.Button('Cubic Map', key="open7"),sg.Text(' x(i) = r * x(i - 1) * (1 - x(i - 1) ^ 2)')],
              [sg.Text('8.'), sg.Button('Variation of Logistic Map', key="open8"),sg.Text('x(i) = r * (1 + x( i - 1)) * (1 + x(i - 1)) * (2 - x(i - 1)) + q')],
              [sg.Text('9.'), sg.Button('Variation of Cheb Map', key="open9"),sg.Text('x(i) = cos(k ^ q * arccos( q * x (i - 1))')],
              [sg.Text('10.'), sg.Button(
                  'Variation of Sine-Sinh Map', key="open10"),sg.Text('x(i) = r * sin(r * sinh(q * sin(2 * x(i - 1))))')],
              [sg.Button('Exit', size=(15, 2)), sg.B('Help', size=(15, 2))]
              ]
    window = sg.Window('Bifurcation diagram',  layout, size=(
        600, 600), resizable=True, finalize=True, grab_anywhere=True, return_keyboard_events=True)
    while True:
        window, event, values = sg.read_all_windows()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Help":
            sg.popup("Choose the map you want to work with and then insert the values, and click the three buttons. Example below:\n\nInitial x = 0 \nq =-0.1 \nInitial r=0 \nEnd of r=1\nStep=0.0001\n\n'q' parameter is only for the variations of the maps.\n\n'Step' are the value you insert divided by the 'End of r'.\nIn the example above the STEP will be 1/0.0001=10000!")
            continue
        if event == "open8":
            extralogistic_window()
            continue
        if event == "open1":
            chebysev_window()
            continue
        if event == "open2":
            sine_sinh_window()
            continue
        if event == "open3":
            logistic_window()
            continue
        if event == "open4":
            sine_window()
            continue
        if event == "open5":
            renyi_window()
            continue
        if event == "open6":
            cubic_logistic_window()
            continue
        if event == "open7":
            cubic_window()
            continue
        if event == "open9":
            extracheb_window()
            continue
        if event == "open10":
            extrasine_sinh_window()
            continue

        window.close()


if __name__ == "__main__":
    main()


# x=r * x * (1 - x) % logistic r- 2*r*x
# x(i)=r*x(i-1)*(1-x(i-1))*(2+x(i-1))%cubic logistic −r*(3*x^2+2*x−2)
# x(i)=r(j)*x(i-1)*(1-x(i-1)^2) %cubic r−3*r*x^2
# x(i)=np.mod(r*x(i-1),1); %renyi r
# x(i)=cos(k*acos(x(i-1))); %cheb r*math.sin(r*math.acos(x))/(math.sqrt(1−x^2))
# x(i)=r(j)*sin(pi*x(i-1)) %sine pi*r*math.cos(pi*x)
# x(i)=k*sin(pi*sinh(pi*sin(pi*x(i-1)))); %sine-sinh pi^3*r*math.cos(pi*x)math.cosh(pi*math.sin(pi*x))math.cos(pi*math.sinh(math.sin(pi*x)))
# math . cos ( k **q * math . acos ( q*x [ i − 1 ] ) ) parallagh cheb
# x[i] = k * math.sin(k * math.sinh(q * math.sin(2 * x[i - 1]))) parallagh np.sine - np.sinh

# x[i] = r * math.sin(r * math.sinh(q * math.sin(2 * x[i - 1]))) # np.sine - np.sinh
# 2*q*r*2*math.cos(2*x)*math.cosh(q*math.sin(2*x))*math.cos(r*math.sinh(q*math.sin(2*x)))

    # x[i] = math.cos(r**q * math.acos(q*x[i - 1])) #cheb

# (q*r**q*math.sin(r**q*math.arccos(q*x)))/(math.sqrt(1-(q**2*x**2)))
