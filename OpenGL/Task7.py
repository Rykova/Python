# -*- coding: utf-8 -*-
# Импортируем все необходимые библиотеки:
import numpy as np
import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True
from OpenGL.GL import *
from OpenGL.GLUT import *
import math
import pywavefront
from OpenGL.GLU import *
from pygame.locals import *
#import sys
# Из модуля random импортируем одноименную функцию random
from random import random
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
    global rP, oldMousePos
    deltaX = x - oldMousePos[0]
    deltaY = y - oldMousePos[1]
    if rP.mouseButton == GLUT_LEFT_BUTTON:
        factor = 0.01
        rP.tVec[0] += deltaX * factor
        rP.tVec[1] -= deltaY * factor
        glTranslatef(deltaX * factor, -deltaY * factor, 0)
        oldMousePos[0], oldMousePos[1] = x, y
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

def drawFirst():
    # Указываем, где взять массив верши:
    # Первый параметр - сколько используется координат на одну вершину
    # Второй параметр - определяем тип данных для каждой координаты вершины
    # Третий парметр - определяет смещение между вершинами в массиве
    # Если вершины идут одна за другой, то смещение 0
    # Четвертый параметр - указатель на первую координату первой вершины в массиве
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    # Указываем, где взять массив цветов:
    # Параметры аналогичны, но указывается массив цветов
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    # Рисуем данные массивов за один проход:
    # Первый параметр - какой тип примитивов использовать (треугольники, точки, линии и др.)
    # Второй параметр - начальный индекс в указанных массивах
    # Третий параметр - количество рисуемых объектов
    glDrawArrays(GL_TRIANGLES, 0, 36)

    pointcolor_core, pointdata_cone = cone();

    glVertexPointer(3, GL_FLOAT, 0, pointdata_cone)
    glColorPointer(3, GL_FLOAT, 0, pointcolor_core)

    glDrawArrays(GL_TRIANGLES, 0, 2160)

def draw_earth():

    glVertexPointer(3, GL_FLOAT, 0, pointdata_earth_new)
    glColorPointer(3, GL_FLOAT, 0, pointcolor_earth)

    glDrawArrays(GL_TRIANGLES, 0, N*3)

# Процедура перерисовки
def draw():
    glClear(GL_COLOR_BUFFER_BIT)                    # Очищаем экран и заливаем серым цветом
    glEnableClientState(GL_VERTEX_ARRAY)            # Включаем использование массива вершин
    glEnableClientState(GL_COLOR_ARRAY)             # Включаем использование массива цветов

    #drawFirst()
    draw_earth()


    glDisableClientState(GL_VERTEX_ARRAY)           # Отключаем использование массива вершин
    glDisableClientState(GL_COLOR_ARRAY)            # Отключаем использование массива цветов
    glutSwapBuffers()                               # Выводим все нарисованное в памяти на экран
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def cone():
    step = 1
    pointdata_cone = np.zeros((6*360,3))
    pointcolor = np.zeros((6*360,3))
    for i in range(360):
        pointdata_cone[3*i,:] = [0,0,1/2]
        pointdata_cone[3*i + 1, :] = [math.sin((i*math.pi)/180), math.cos((i*math.pi)/180), 0]
        pointdata_cone[3*i + 2, :] = [math.sin(((i+step)*math.pi)/180), math.cos(((i+step)*math.pi)/180), 0]
        pointcolor[3 * i, :] = [1.0,1.0,0.0]
        pointcolor[3 * i + 1, :] = [1.0, 1.0, 0.0]
        pointcolor[3 * i + 2, :] = [1.0, 1.0, 0.0]

    for i in range(360, 2*360):
        pointdata_cone[3 * i, :] = [1/2, 0, 0]
        pointdata_cone[3 * i + 1, :] = [math.cos((i*math.pi)/180), math.sin((i*math.pi)/180), 0]
        pointdata_cone[3 * i + 2, :] = [math.cos(((i+step)*math.pi)/180), math.sin(((i+step)*math.pi)/180), 0]
        pointcolor[3 * i, :] = [1.0, 0.0, 0.0]
        pointcolor[3 * i + 1, :] = [1.0, 0.0, 0.0]
        pointcolor[3 * i + 2, :] = [1.0, 0.0, 0.0]

    return  pointcolor, pointdata_cone

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
# Определяем массив вершин (три вершины по три координаты)
pointdata = [[-1.0,-1.0,-1.0],
    [-1.0,-1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0,-1.0],
    [-1.0,-1.0,-1.0],
    [-1.0, 1.0,-1.0],
    [1.0,-1.0, 1.0],
    [-1.0,-1.0,-1.0],
    [1.0,-1.0,-1.0],
    [1.0, 1.0,-1.0],
    [1.0,-1.0,-1.0],
    [-1.0,-1.0,-1.0],
    [-1.0,-1.0,-1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0,-1.0],
    [1.0,-1.0, 1.0],
    [-1.0,-1.0, 1.0],
    [-1.0,-1.0,-1.0],
    [-1.0, 1.0, 1.0],
    [-1.0,-1.0, 1.0],
    [1.0,-1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0,-1.0,-1.0],
    [1.0, 1.0,-1.0],
    [1.0,-1.0,-1.0],
    [1.0, 1.0, 1.0],
    [1.0,-1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0,-1.0],
    [-1.0, 1.0,-1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0,-1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0,-1.0, 1.0]]
# Определяем массив цветов (по одному цвету для каждой вершины)
pointcolor =   [
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1,  0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    ]
pointdata = np.add(pointdata,[0,0,-1])
pointdata = np.divide(pointdata,4)
# Запускаем основной цикл

#window = pyglet.window.Window(1024, 720, caption = 'Demo', resizable = True)
name = 'earth.obj'
meshes = pywavefront.Wavefront(name)
ps = pywavefront.ObjParser(meshes, name)
ps.read_file(name)
pointdata2 = ps.material.vertices
N = len(pointdata2) // 24
pointdata_earth = np.zeros((N, 3, 3))
pointcolor_earth = np.zeros((N * 3, 3))

for i in range(0, N):
    for j in range(0, 3):
        pointdata_earth[i, j, 0:3] = pointdata2[24 * i + 8 * j + 5:24 * i + 8 * j + 8]
pointdata_earth /= pointdata_earth.max()
pointdata_earth_new = np.zeros((N * 3, 3))
for i in range(0, N):
    pointcolor_earth[3 * i, :] = [1.0, 0.5, 0.0]
    pointcolor_earth[3 * i + 1, :] = [0.5, 1.0, 0.0]
    pointcolor_earth[3 * i + 2, :] = [1.0, 1.0, 0.0]
    pointdata_earth_new[3 * i, :] = pointdata_earth[i, 0, 0:3]
    pointdata_earth_new[3 * i + 1, :] = pointdata_earth[i, 1, 0:3]
    pointdata_earth_new[3 * i + 2, :] = pointdata_earth[i, 2, 0:3]

#pyglet.clock.schedule(update)

#pyglet.app.run()
glRotatef(65, 2, 1, 0)

glutMainLoop()
