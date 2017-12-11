
import matplotlib
import numpy as np
import xml.etree.cElementTree as ET

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import numpy.linalg as nlg
import multiprocessing as mp

from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import *
from tkinter import filedialog


import TasksOfNBodiesFunction

entry = {}

LARGE_FONT = ("Verdana", 20)

MassSun = 1.99 * pow(10, 30)
MassEarth = 5.98 * pow(10, 24)
MassMoon = 7.32 * pow(10, 22)
MassMerc = 3.285 *pow(10,23)
M = 500
T = 5*12 * 2592000
r_norm = 1.496 * 10 ** 11

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

    root = ET.Element("root")
    params = ET.SubElement(root, "parameters")

    SizeX = ET.SubElement(params, 'SizeAreaX')
    SizeX.text = str(entry['SizeAreaX'])
    SizeY = ET.SubElement(params, 'SizeAreaY')
    SizeY.text = str(entry['SizeAreaY'])
    Obj = ET.SubElement(params, 'Objects')
    Obj.text =  str(entry['Objects'])
    Col = ET.SubElement(params,'Color')
    Col.text = str(entry['Color'])
    Slider = ET.SubElement(params,'SizeSlider')
    Slider.text = str(entry['SizeSlider'])

    circles = ET.SubElement(root, "Objects")
    for cir in entry["Objects"]:
        circle = ET.SubElement(circles, "Object")
        ET.SubElement(circle, "x").text = str(cir.x)
        ET.SubElement(circle, "y").text = str(cir.y)
        ET.SubElement(circle, "size").text = str(cir.size)
        ET.SubElement(circle, "color").text = str(cir.color)

    tree = ET.ElementTree(root)
    tree.write(f.name)

    #with open(f.name, 'wb') as handle:
     #   pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)

class SeaofBTCapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)


        tk.Tk.wm_title(self, "My interface")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        F = GraphPage
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
        container = ttk.Frame(nb)
        page1 = PageModel(nb,container)


        nb.add(page2, text='Edit')
        nb.add(page1, text='Model')

        nb.pack(expand=1, fill="both")

        #########################################

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


        #with open(filename, 'rb') as file:
        #    entry = pickle.load(file)

        #self.ax.set_xlim(entry['SizeAreaX'])
        #self.ax.set_ylim(entry['SizeAreaY'])

        #self.labelSlider.delete(0, tk.END)
        #self.labelSlider.insert(0, str(entry['SizeSlider']))

        #self.Slider.set(entry['SizeSlider'])
        #self.curr_color = entry['Color']
        #self.SizeAreaY = entry['SizeAreaY'];
        #self.SizeAreaX = entry['SizeAreaX'];
        #self.curr_size = entry['SizeSlider'];

        #self.ax.axes.clear()
        #self.circles.clear()

        #for elem in entry['Objects']:
         #   circle = customCircle(elem.x, elem.y, elem.size, elem.color)
         #   self.ax.axes.add_artist(circle)
         #   self.circles.append(circle)

        #self.f.canvas.draw()

        tree = ET.parse(filename)
        parameters = tree.find("parameters")
        plot_size = str(parameters.find('SizeAreaX').text)
        spl = plot_size.split('(')[1].split(')')[0].split(',')
        x_lim = (float(spl[0]), float(spl[1]))
        plot_size = str(parameters.find('SizeAreaY').text)
        spl = plot_size.split('(')[1].split(')')[0].split(',')
        y_lim = (float(spl[0]), float(spl[1]))

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)


        self.labelSlider.delete(0, tk.END)
        self.labelSlider.insert(0, str(parameters.find('SizeSlider').text))

        self.Slider.set(float(parameters.find('SizeSlider').text))

        self.curr_color = str(parameters.find('Color').text)
        self.SizeAreaY = x_lim
        self.SizeAreaX =  y_lim
        self.curr_size = float(parameters.find('SizeSlider').text)

        self.ax.axes.clear()
        self.circles.clear()

        self.circles.clear()
        circles_root = tree.find("Objects")
        circles = circles_root.getiterator("Object")

        for c in circles:
            x = c.find("x").text
            y = c.find("y").text
            size = c.find("size").text
            color = c.find("color").text
            newCircle = customCircle(float(x), float(y), float(size), color)
            self.circles.append(newCircle)
            self.ax.axes.add_artist(newCircle)

        self.f.canvas.draw()

        #filename_xsd = "XSD.xsd";
        #with open(filename_xsd, 'r') as schema_file:
        #    schema_to_check = schema_file.read()
        #with open(filename, 'r') as xml_file:
        #    xml_to_check = xml_file.read()

        #xmlschema_doc = tree.parse(StringIO(schema_to_check))
        #xmlschema = tree.XMLSchema(xmlschema_doc)
        #try:
        #    doc = tree.parse(StringIO(xml_to_check))
        #    print('XML well formed, syntax ok.')
        #except IOError:
        #    print('Invalid File')

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

    def MethodVerle(self, x, y):
        init = x,y
        r2 = np.sqrt(x) + np.sqrt(y)
        r = np.sqrt(r2)

class customCircle(Circle):
    def __init__(self, x, y, size, color):
        Circle.__init__(self, (x, y), size)
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.set_color(color)

class PageModel(ttk.Frame):


    def __init__(self, parent, container):
        ttk.Frame.__init__(self, parent)
        MODES = [
            ("scipy",1), ("verlet",2),
            ("verlet-threading",3), ("verlet-multiprocessing",4),
            ("verlet-cython without typed memoryview",5),("verlet-cython with typed memoryview",6), ("verlet-openmp without typed memoryview",7),
            ("verlet-openmp with typed memoryview",8),("verlet-opencl",9)
        ]

        v = StringVar()
        v.set("L")  # initialize

        def getScript():
            state = v.get()
            if(state == '1'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("scipy")

            if(state == '2'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet")


            if(state == '3'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-threading")

            if (state == '4'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-multiprocessing")

            if(state == '5'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-cython without typed memoryview")

            if(state == '6'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-cython with typed memoryview")

            if (state == '7'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-openmp without typed memoryview")

            if (state == '8'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-openmp with typed memoryview")

            if (state == '9'):
                result = TasksOfNBodiesFunction.TaskOfNbodiesVerle("verlet-opencl")


            ax.axes.clear()
            x_lim_0 = np.min(result[:, :, 0])
            y_lim_0 = np.min(result[:, :, 1])
            x_lim_1 = np.max(result[:, :, 0])
            y_lim_1 = np.max(result[:, :, 1])

            ax.set_xlim(x_lim_0, x_lim_1)
            ax.set_ylim(y_lim_0, y_lim_1)

            for i in range(0, M):
                ax.axes.clear()
                ax.set_xlim(x_lim_0, x_lim_1)
                ax.set_ylim(y_lim_0, y_lim_1)
                x1 = result[i, 0, 0]
                y1 = result[i, 0, 1]
                x2 = result[i, 1, 0]
                y2 = result[i, 1, 1]
                x3 = result[i, 2, 0]
                y3 = result[i, 2, 1]
                x4 = result[i, 3, 0]
                y4 = result[i, 3, 1]
                newCircle = customCircle(float(x1), float(y1), 10 ** 5 * MassMoon / MassSun, "green")
                ax.axes.add_artist(newCircle)
                newCircle = customCircle(float(x2), float(y2), 10 ** 3 * MassEarth / MassSun, "blue")
                ax.axes.add_artist(newCircle)
                newCircle = customCircle(float(x3), float(y3), 10 ** (-1), "yellow")
                ax.axes.add_artist(newCircle)
                newCircle = customCircle(float(x4), float(y4), 10 ** 5 * MassMerc / MassSun, "red")
                ax.axes.add_artist(newCircle)
                canvas_model.draw()

        #Calculate_Defect()
        #Get_Average_Time()
        #Time_of_all_methods()



        for text, mode in MODES:
            b = Radiobutton(self, text=text,
                            variable=v, value=mode,command = getScript)
            b.pack(anchor=W)


        f = Figure(figsize=(5, 5), dpi=100)
        ax = f.add_subplot(111)
        canvas_model = FigureCanvasTkAgg(f, self)
        canvas_model.show()
        canvas_model.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas_model._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_model = NavigationToolbar2TkAgg(canvas_model, self)
        toolbar_model.update()

if __name__ == '__main__':

    app = SeaofBTCapp()
    menubar = Menu(app)
    text = Text(app)
    text.pack()
    filemenu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=filemenu)

    filemenu.add_command(label="Save", command=file_save)

    app.config(menu=menubar)
    app.mainloop()


