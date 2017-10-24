import sys
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import *
from tkinter import filedialog

import pickle
entry = {}

LARGE_FONT = ("Verdana", 12)


def isConvertibleToFloat(value):
    try:
        float(value)
        return True
    except:
        return False

def file_save():
    f = filedialog.asksaveasfile(mode='w', defaultextension=".xml")
    if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    f.close()  # `()` was missing.
    with open(f.name, 'wb') as handle:
        pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)



class SeaofBTCapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)


        tk.Tk.wm_title(self, "My interface")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        F = GraphPage;
        frame = F(container, self)

        self.frames[F] = frame

        frame.grid(row=0, column=0, sticky="nsew")


    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class GraphPage(tk.Frame):

    curr_size = 1.0
    curr_color = ((115.44921875, 206.8046875, 124.484375), '#73ce7c')
    # list with circles
    circles = []
    SizeAreaX = (-100,100)
    SizeAreaY = (-100,100)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Main Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        nb = ttk.Notebook(self)

        # adding Frames as pages for the ttk.Notebook
        # first page, which would get widgets gridded into it
        page2 = ttk.Frame(nb)

        # second page
        page1 = ttk.Frame(nb)

        nb.add(page2, text='Edit')
        nb.add(page1, text='Model')

        nb.pack(expand=1, fill="both")

        self.f = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.f.add_subplot(111)


        self.ax.set_xlim(self.SizeAreaX)
        self.ax.set_ylim(self.SizeAreaY)

        entry['SizeAreaX'] = self.SizeAreaX
        entry['SizeAreaY'] = self.SizeAreaY
        entry['SizeSlider'] = self.curr_size
        entry['Objects'] = self.circles
        entry['Color'] = self.curr_color



        def IncreaseGraph():
            y_lim = self.ax.axes.get_ylim()
            x_lim = self.ax.axes.get_xlim()

            self.ax.set_xlim(np.multiply(x_lim, 1.5))
            self.ax.set_ylim(np.multiply(y_lim, 1.5))

            entry['SizeAreaX'] = np.multiply(x_lim, 1.5)
            entry['SizeAreaY'] = np.multiply(y_lim, 1.5)

            canvas_edit.draw()

        def DecreaseGraph():
            y_lim = self.ax.axes.get_ylim()
            x_lim = self.ax.axes.get_xlim()

            self.ax.set_xlim(np.divide(x_lim, 1.5))
            self.ax.set_ylim(np.divide(y_lim, 1.5))

            entry['SizeAreaX'] = np.divide(x_lim, 1.5)
            entry['SizeAreaY'] = np.divide(y_lim, 1.5)

            canvas_edit.draw()

        buttonOpen = ttk.Button(page2,text="Open", command=self.file_open)
        buttonOpen.pack()

        buttonPlus = ttk.Button(page2, text="-",
                             command=IncreaseGraph)
        buttonPlus.pack()

        buttonMinus = ttk.Button(page2, text="+",
                                command=DecreaseGraph)

        buttonMinus.pack()


        self.curr_x = 0
        self.curr_y = 0

        self.labelx = tk.Label(page2, text="Coordinate x = ", font=LARGE_FONT)
        self.labelx.pack()

        self.labely = tk.Label(page2, text="Coordinate y = ", font=LARGE_FONT)
        self.labely.pack()

        self.buttonColor = ttk.Button(page2,text='Select Color', command=self.getColor)
        self.buttonColor.pack()


        self.v = tk.StringVar(page2, value='1.0')
        self.v.trace("w", lambda name, index, mode, v=self.v: self.ChangeText(v))
        self.labelSlider = tk.Entry(page2,textvariable= self.v)
        self.labelSlider.pack()
        self.Slider = ttk.Scale(page2, from_=1.0, to=100.0,command=self.ChangeSize)
        self.Slider.pack()



        canvas_edit = FigureCanvasTkAgg(self.f, page2)
        canvas_edit.show()
        canvas_edit.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas_edit._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_edit = NavigationToolbar2TkAgg(canvas_edit, page2)
        toolbar_edit.update()

        self.f.canvas.mpl_connect('motion_notify_event', self.changeCoords)
        self.f.canvas.mpl_connect('button_press_event', self.drawCircle)

    def file_open(self):
        filename = filedialog.askopenfilename()
        if filename == "":
            return
        if filename is None:  # return `None` if dialog closed with "cancel".
            return
        with open(filename, 'rb') as file:
            entry = pickle.load(file)

        self.ax.set_xlim(entry['SizeAreaX'])
        self.ax.set_ylim(entry['SizeAreaY'])

        self.labelSlider.delete(0, tk.END)
        self.labelSlider.insert(0, str(entry['SizeSlider']))

        self.Slider.set(entry['SizeSlider'])
        self.curr_color = entry['Color']
        self.SizeAreaY = entry['SizeAreaY'];
        self.SizeAreaX = entry['SizeAreaX'];
        self.curr_size = entry['SizeSlider'];

        self.ax.axes.clear()
        self.circles.clear()

        for elem in entry['Objects']:
            circle = customCircle(elem.x, elem.y, elem.size, elem.color)
            self.ax.axes.add_artist(circle)
            self.circles.append(circle)
        self.f.canvas.draw()

    def changeCoords(self, event):
        if (event.inaxes):
            self.curr_x = event.xdata
            self.curr_y = event.ydata
            self.labelx.config(text="Coordinate x = " + str(event.xdata))
            self.labely.config(text="Coordinate y = " + str(event.ydata))

    def ChangeText(self,v):
        print(v.get())
        if isConvertibleToFloat(v.get()):
            val = float(v.get())
            self.Slider.set(val)

    def ChangeSize(self,event):
            self.curr_size = self.Slider.get()
            entry['SizeSlider'] = self.curr_size
            self.labelSlider.delete(0, tk.END)
            self.labelSlider.insert(0, str(self.curr_size))


    def drawCircle(self, event):
        if (event.inaxes):
            circle = customCircle(self.curr_x, self.curr_y, self.curr_size, self.curr_color[1])
            self.ax.axes.add_artist(circle)
            self.circles.append(circle)
            entry['Objects'] = self.circles
            self.f.canvas.draw()

    def getColor(self):
        self.curr_color = askcolor()
        entry['Color'] = self.curr_color



class customCircle(Circle):
    def __init__(self, x, y, size, color):
        Circle.__init__(self, (x, y), size)
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.set_color(color)





app = SeaofBTCapp()
menubar = Menu(app)
text = Text(app)
text.pack()
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)

filemenu.add_command(label="Save", command=file_save)


app.config(menu=menubar)
app.mainloop()
