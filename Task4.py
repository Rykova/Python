
import matplotlib
import numpy as np
import copy
import numpy.linalg as nlg
import threading
import time


matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt



from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import *
from tkinter import filedialog


from scipy.integrate import odeint
import pickle
entry = {}

LARGE_FONT = ("Verdana", 20)


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

def TaskOfNbodiesVerle(type):
    MassSun =  1.99*pow(10, 30)
    MassEarth = 5.98 * pow(10, 24)
    MassMoon = 7.32 * pow(10, 22)

    list_of_mass_all = [MassMoon, MassEarth, MassSun]
    #list_of_mass_all = [MassMoon/MassSun, MassEarth/MassSun, 1]
    list_of_radius_and_velocity_all = np.zeros((3, 6))

    M = 12*pow(10,6)
    T = 12*2592000

    #first body moon
    list_of_radius_velocity = [0, -384467000, 0, 1022, -29.783*10 **3, 0]
    list_of_radius_and_velocity_all[0,:] = list_of_radius_velocity

    #second body earth
    list_of_radius_velocity = [0, 0, 0, 0, -29.783*10 **3, 0]
    list_of_radius_and_velocity_all[1, :] = list_of_radius_velocity

    # third sun
    list_of_radius_velocity = [1.496*10 ** 11, 0, 0, 0, 0, 0]
    list_of_radius_and_velocity_all[2, :] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init =list_of_radius_and_velocity_all.reshape((6 * N))


    if(type == "verlet"):
        result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return  result
    if (type =="scipy"):
        time_span = np.linspace(0,T,M)
        result = odeint(g,init,time_span,args=(list_of_mass_all,N))
        result2 = result.reshape((M, N, 6))
        return result2
    if (type == "verlet-threading"):
        result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result

def Velocity_form_for_v(list,list_new,a,a_new,tau,j):
    list_new[j,3:6]=list[j,3:6]+0.5*(a+a_new)*tau

def Velocity_form_for_x(list,list_new,a,tau,j):
    list_new[j,0:3]=list[j,0:3]+list[j,3:6]*tau+0.5*a*tau**2

def accelaration_for_i_body(a, N, list_of_data,list_of_mass,i):
    G=6.67 * 10 **(-11) #gravitation const
    for j in range(0,N):
        if i!=j:
            a[i]+=G*list_of_mass[j]*(list_of_data[j,0:3]-list_of_data[i,0:3])/nlg.norm(list_of_data[j,0:3]-list_of_data[i,0:3],2) ** 3

def VerletMethod(list_of_radius_and_velocity, list_of_mass_all, N, M, T):
    tau=T/M
    N = len(list_of_radius_and_velocity)
    print(N)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    #acceleration
    a = np.zeros((N, 3))
    for i in range(0, N):
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, i)

    for i in range(1,M):
        list_of_radius_and_velocity_new = np.zeros((N,6))
        for j in range(0,N):
            Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)
        a_new = np.zeros((N, 3))
        for k in range(0, N):
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, k)

        for j in range(0,N):
            Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)

        list_of_radius_and_velocity=copy.copy(list_of_radius_and_velocity_new)
        a=copy.copy(a_new)
        result[i]=copy.copy(list_of_radius_and_velocity)
    print(a)


    return result


def ThreadingWork(M,N,list_of_thread,th_ev):
    for i in range(1,M):
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()


def ThreadMethod(result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev):

    N=len(list_of_radius_and_velocity)
    a = np.zeros((N, 3))
    accelaration_for_i_body(a, N, list_of_radius_and_velocity,list_of_mass_all,j)
    for i in range(1,M):
        list_of_radius_and_velocity_new[j]=np.zeros(6)
        Velocity_form_for_x(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],tau,j)
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()
        a_new = np.zeros((N, 3))
        accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, j)
        Velocity_form_for_v(list_of_radius_and_velocity,list_of_radius_and_velocity_new,a[j],a_new[j],tau,j)
        list_of_radius_and_velocity[j]=copy.copy(list_of_radius_and_velocity_new[j])
        a=copy.copy(a_new)
        result[i,j]=copy.copy(list_of_radius_and_velocity[j])
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()



def VerletMethodThreading(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau=T/M
    N = len(list_of_radius_and_velocity)
    print(N)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    th_ev = threading.Event()
    list_of_radius_and_velocity_new = np.zeros((N, 6))
    list_of_thread = []
    for j in range(0, N):
        th = threading.Event()
        list_of_thread.append(th)
    Threads = threading.Thread(target=ThreadingWork, name="ThreadingWork", args=(M, N, list_of_thread, th_ev))
    Threads.start()
    for j in range(0, N):
        t = threading.Thread(target=ThreadMethod, name="thread" + str(j), args=(result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev))
        t.start()
    Threads.join()

    return result





def g(list_of_data,time_span,list_of_mass,N):
    G = 6.67 * 10 ** (-11)
    mass_of_funct = np.zeros(6*N)
    for i in range(0, N):
        f1 = list_of_data[6 * i + 3 : 6 * i + 6]
        f2 = np.zeros(3)
        for j in range(0, N):
            if (i != j):
                f2 += G * list_of_mass[j] * (list_of_data[6*j:6*j+3] - list_of_data[6*i:6*i+3]) / nlg.norm(list_of_data[6*j:6*j+3] - list_of_data[6*i:6*i+3], 2) ** 3
        mass_of_funct[6*i:6*i+3] = f1
        mass_of_funct[6*i+3:6*i+6] = f2
    return mass_of_funct


class PageModel(ttk.Frame):


    def __init__(self, parent, container):
        ttk.Frame.__init__(self, parent)
        MODES = [
            ("scipy",1), ("verlet",2),
            ("verlet-threading",3), ("verlet-multiprocessing",4),
            ("verlet-cython",5), ("verlet-opencl",6),
        ]

        v = StringVar()
        v.set("L")  # initialize

        def getScript():
            state = v.get()
            if(state == '1'):
                t = time.time()
                result = TaskOfNbodiesVerle("scipy")
                print(time.time() - t)
                x1 = result[:, 0, 0]
                y1 = result[:, 0, 1]
                x2 = result[:, 1, 0]
                y2 = result[:, 1, 1]
                x3 = result[:, 2, 0]
                y3 = result[:, 2, 1]
                ax.axes.clear()
                ax.plot(x1, y1)
                ax.plot(x2, y2, color="red")
                ax.plot(x3, y3, color="green")
                canvas_model.draw()

            if(state == '2'):
                t = time.time()
                result = TaskOfNbodiesVerle("verlet")
                print(time.time() - t)
                x1 = result[:, 0, 0]
                y1 = result[:, 0, 1]
                x2 = result[:, 1, 0]
                y2 = result[:, 1, 1]
                x3 = result[:, 2, 0]
                y3 = result[:, 2, 1]
                ax.axes.clear()
                ax.plot(x1, y1,color="blue")
                ax.plot(x2, y2, color="red")
                ax.plot(x3, y3, color="green")
                canvas_model.draw()
                plt.plot(x1, y1, color="blue")
                plt.plot(x2, y2, color="red")
                plt.plot(x3, y3, color="green")
                plt.show()
            if(state == '3'):
                t = time.time()
                result = TaskOfNbodiesVerle("verlet-threading")
                print(time.time() - t)
                print(result)
                x1 = result[:, 0, 0]
                y1 = result[:, 0, 1]
                x2 = result[:, 1, 0]
                y2 = result[:, 1, 1]
                ax.axes.clear()
                ax.plot(x1, y1)
                ax.plot(x2, y2, color="red")
                canvas_model.draw()

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


app = SeaofBTCapp()
menubar = Menu(app)
text = Text(app)
text.pack()
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)

filemenu.add_command(label="Save", command=file_save)


app.config(menu=menubar)
app.mainloop()
