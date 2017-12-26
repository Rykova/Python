# -*- coding: utf-8 -*-
# Импортируем все необходимые библиотеки:
import numpy as np
import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True
from OpenGL.GL import *
from OpenGL.GLUT import *
import math
import pywavefront
import scipy.optimize as so
import scipy.integrate as si
import copy
from random import random

class HeatSolver:
    def __init__(self, lambada, Q_R, c, eps, S, tau):
        self.lambada=lambada
        self.Q_R=Q_R
        self.c=c
        self.M=len(self.c)
        self.eps=eps
        self.S=S
        self.counter=0
        self.T0=so.fsolve(func_solve,np.zeros(self.M),args=(0,lambada,Q_R,c,eps,S,))
        self.T=copy.copy(self.T0)
        self.tau=tau
    def next_step(self):
        Tm=np.linspace((self.counter-1)*self.tau,self.counter*self.tau,2)
        self.counter+=1
        self.T=si.odeint(func_solve,self.T0,Tm,args=(self.lambada,self.Q_R,self.c,self.eps,self.S,))
        self.T0=copy.copy(self.T[1])
        return self.T[1]
def func_solve(T,t,lambada, Q_R, c, eps, S):
    M=len(c)
    right_part=np.zeros(M)
    C0=5.67
    for i in range(M):
        for j in range(M):
            if i!=j:
                right_part[i]-=lambada[i,j]*S[i,j]*(T[i]-T[j])
        right_part[i]-=eps[i]*S[i,i]*C0*(T[i]/100)**4
        right_part[i]+=Q_R[i](t)
        right_part[i]/=c[i]
    return right_part

# объявляем массив pointcolor глобальным (будет доступен во всей программе)
def setList( l, v ):
	"""Set all elements of a list to the same bvalue"""
	for i in range( len( l ) ):
		l[i] = v
class renderParam( object ):
	"""Class holding current parameters for rendering.

	Parameters are modified by user interaction"""
	def __init__( self ):
		self.initialColor = [1, 1, 1]
		self.drawColor = self.initialColor
		self.tVec = [0, 0, 0]
		self.mouseButton = None

	def reset( self ):
		self.drawColor = self.initialColor
		setList( self.tVec, 0 )
		self.mouseButton = None

global pointcolor
oldMousePos = [0,0]
rP = renderParam()
def mouseButton(button,mode,x,y):
    global rP, oldMousePos
    if mode == GLUT_DOWN:
        rP.mouseButton = button
    else:
        rP.mouseButton = None
    oldMousePos[0], oldMousePos[1] = x, y
    glutPostRedisplay()
def mouseMotion(x,y):
    glMatrixMode(GL_PROJECTION)
    global rP, oldMousePos
    deltaX = x - oldMousePos[0]
    deltaY = y - oldMousePos[1]
    if rP.mouseButton == GLUT_LEFT_BUTTON:
        factor = 0.01
        rP.tVec[0] += deltaX * factor
        rP.tVec[1] -= deltaY * factor
        glTranslatef(deltaX * factor, -deltaY * factor, 0)
        oldMousePos[0], oldMousePos[1] = x, y
    glMatrixMode(GL_MODELVIEW)
    glutPostRedisplay()


def registerCallbacks():
    glutMouseFunc(mouseButton)
    glutMotionFunc(mouseMotion)
    glutSpecialFunc(specialkeys)

# Процедура обработки специальных клавиш
def specialkeys(key, x, y):
    # Сообщаем о необходимости использовать глобального массива pointcolor
    global pointcolor


    # Обработчики специальных клавиш
    if key == GLUT_KEY_UP:          # Клавиша вверх
        glRotatef(5, 1, 0, 0)
    if key == GLUT_KEY_DOWN:        # Клавиша вниз
        glRotatef(-5, 1, 0, 0)
    if key == GLUT_KEY_LEFT:        # Клавиша влево
        glRotatef(5, 0, 1, 0)
    if key == GLUT_KEY_RIGHT:       # Клавиша вправо
        glRotatef(-5, 0, 1, 0)
    if key == GLUT_KEY_F1:
        for j in range(0,30):
            for i in range(len(list_of_vel)):
                Col = From_heat_to_color(slv.T0[i])
                for j in range(diapazon[i], diapazon[i + 1]):
                    pointcolor[j] = [Col, Col, Col]
            slv.next_step()
        print(slv.T0)


# Процедура подготовки шейдера (тип шейдера, текст шейдера)
def create_shader(shader_type, source):
    # Создаем пустой объект шейдера
    shader = glCreateShader(shader_type)
    # Привязываем текст шейдера к пустому объекту шейдера
    glShaderSource(shader, source)
    # Компилируем шейдер
    glCompileShader(shader)
    # Возвращаем созданный шейдер
    return shader

def Form_Triangles(list_of_vel,list_of_tri):
    triangles = []
    diapazon = np.zeros(len(list_of_vel) + 1, dtype=np.int)
    for i in range(len(list_of_vel)):
        diapazon[i + 1] = diapazon[i] + len(list_of_tri[i])
        for el in list_of_tri[i]:
            triangle = np.array([list_of_vel[i][int(el[0])], list_of_vel[i][int(el[1])], list_of_vel[i][int(el[2])]])
            triangles.append(triangle)
    return np.array(triangles), diapazon

def From_heat_to_color(Te):
    B=500
    D=500
    if Te<50:
        return [np.exp(-(Te-50)**2/B), np.exp(-(Te)**2/B),np.exp(-(Te+50)**2/B)]
    return [max(np.exp(-(Te-50)**2/B),np.exp(-(Te-100)**2/D)), np.exp(-(Te-100)**2/D),np.exp(-(Te-100)**2/D)]


# Процедура перерисовки
def draw():
    glClear(GL_COLOR_BUFFER_BIT)                    # Очищаем экран и заливаем серым цветом
    glEnableClientState(GL_VERTEX_ARRAY)            # Включаем использование массива вершин
    glEnableClientState(GL_COLOR_ARRAY)             # Включаем использование массива цветов

    glVertexPointer(3, GL_FLOAT, 0, triangles)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)

    glDrawArrays(GL_TRIANGLES, 0, 3 * diapazon[len(list_of_vel)])

    glDisableClientState(GL_VERTEX_ARRAY)           # Отключаем использование массива вершин
    glDisableClientState(GL_COLOR_ARRAY)            # Отключаем использование массива цветов
    glutSwapBuffers()                               # Выводим все нарисованное в памяти на экран
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def parseFile(file_name):
    count = 0
    start = 0
    list_of_count_of_vel = []
    list_of_vel = []
    list_of_triang = []
    index = 0
    lst_vel = []
    lst_f = []
    total = 1
    count_of_v = 0

    for line in open(file_name, 'r'):
        values = line.split()
        if len(values) < 2:
            continue
        if(values[0]== '#' and values[1] == 'object' and count != 0):
            list_of_count_of_vel.append(count)
            list_of_vel.append(lst_vel)
            list_of_triang.append(lst_f)
            index = index + 1
            total = total + count_of_v
            count_of_v = 0
            count = 0
            lst_vel = []
            lst_f = []
        if (values[0] == '#' and values[1] == 'object' and count == 0):
            start = 1

        if(values[0] == 'f' and count == 0):
            start = 1

        if(start == 1 and values[0] == 'f'):
            count = count + 1
            lst_f.append([float(values[1])-total,float(values[2])-total,float(values[3])-total])

        if (start == 1 and values[0] == 'v'):
            lst_vel.append([float(values[1]), float(values[2]), float(values[3])])
            count_of_v = count_of_v + 1

    list_of_vel.append(lst_vel)
    list_of_triang.append(lst_f)
    list_of_count_of_vel.append(count)
    return  list_of_count_of_vel,list_of_triang,list_of_vel

# Здесь начинется выполнение программы
# Использовать двойную буферезацию и цвета в формате RGB (Красный Синий Зеленый)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
# Указываем начальный размер окна (ширина, высота)
glutInitWindowSize(600, 600)
# Указываем начальное
# положение окна относительно левого верхнего угла экрана
glutInitWindowPosition(50, 50)

# Инициализация OpenGl
glutInit(sys.argv)
# Создаем окно с заголовком "Shaders!"
glutCreateWindow(b"Shaders!")
# Определяем процедуру, отвечающую за перерисовку
glutDisplayFunc(draw)
# Определяем процедуру, выполняющуюся при "простое" программы
glutIdleFunc(draw)
# Определяем процедуру, отвечающую за обработку клавиш
#glutSpecialFunc(specialkeys)
registerCallbacks()
# Задаем серый цвет для очистки экрана
glClearColor(0.2, 0.2, 0.2, 1)
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)
# Создаем вершинный шейдер:
# Положение вершин не меняется
# Цвет вершины - такой же как и в массиве цветов
vertex = create_shader(GL_VERTEX_SHADER, """
varying vec4 vertex_color;
            void main(){
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                vertex_color = gl_Color;
           }""")
# Создаем фрагментный шейдер:
# Определяет цвет каждого фрагмента как "смешанный" цвет его вершин
fragment = create_shader(GL_FRAGMENT_SHADER, """
varying vec4 vertex_color;
            void main() {
                gl_FragColor = vertex_color;
}""")
# Создаем пустой объект шейдерной программы
program = glCreateProgram()
# Приcоединяем вершинный шейдер к программе
glAttachShader(program, vertex)
# Присоединяем фрагментный шейдер к программе
glAttachShader(program, fragment)
# "Собираем" шейдерную программу
glLinkProgram(program)
# Сообщаем OpenGL о необходимости использовать данную шейдерну программу при отрисовке объектов
glUseProgram(program)


list,list_of_tri,list_of_vel = parseFile('model1.obj')
M = len(list)
triangles, diapazon=Form_Triangles(list_of_vel,list_of_tri)
triangles/=(2*triangles.max())
pointcolor=np.zeros((diapazon[len(list_of_vel)],3,3))
for i in range(0, len(list)):
    m = random()
    k = random()
    for j in range(diapazon[i],diapazon[i+1]):
        pointcolor[j] = [k,m,0.0]

#Значения коэффициентов
square=np.loadtxt('square.txt')
for i in range(M):
    for j in range(i+1,M):
        temp=(square[i,j]+square[j,i])/2
        square[i,j]=temp
        square[j,i]=temp

eps = [0.05,0.05,0.05,0.01,0.1]
c = [900,900,900,840,520]

lambada=np.zeros((M,M))
lambada[0,1]=240
lambada[1,0]=240
lambada[1,2]=240
lambada[2,1]=240
lambada[2,3]=119
lambada[3,2]=119
lambada[3,4]=10.5
lambada[4,3]=10.5

Q_R=[]
for i in range(M):
    f=lambda t: [0]
    Q_R.append(f)
A=2
Q_R[2]=lambda t:[A*(20+3*np.sin(t/4))]
tau=10**2
slv=HeatSolver(lambada,Q_R,c,eps,square,tau)
for i in range(len(list_of_vel)):
    Col = From_heat_to_color(slv.T0[i])
    for j in range(diapazon[i], diapazon[i + 1]):
        pointcolor[j] = [Col, Col, Col]

glRotatef(65, 2, 1, 0)
glutMainLoop()

