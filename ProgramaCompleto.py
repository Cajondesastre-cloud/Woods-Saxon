# Código del programa empleado para obtener resultados en el TFG del grado
# en Física de la Universidad de Alicante: Estudio numérico del potencial
# de Woods-Saxon. Elaborado por David Montiel López (dml22@alu.ua.es).

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation

"""Integración de la resolución de la ecuación de Schrödinger usando diversos
métodos para ecuaciones diferenciales y de obtención de autovalores. Se puede
optar por emplear el menú para seleccionar las distintas opciones, o pasar
todos los parámetros de la simulación en un array."""

# Parámetros simulación

A = 56                                     # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial principal en MeV.
R = r0*A**(1/3)                            # Constante radial corteza.
Rn = 15                                    # Límite array.
n = 0                                      # Inicialización número principal.
l = 0                                      # Inicialización número azimutal.
x = np.linspace(0, R + Rn, 1000)           # Array de posiciones.
dx = x[1]-x[0]                             # Diferencial de x.
W0 = 50                                    # Inicialización pozo en MeV.


# Funciones de resolución de ecuaciones diferenciales.

def Runge4(h,f,g,w,u,r, E): # Método de Runge-Kutta Orden 4 para ec. dif 2º ord.
    
    k1 = h*f(w,u,r,E)
    k1_g = h*g(w,u,r, E)
    k2 = h*f(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    k2_g = h*g(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    k3 = h*f(w + 1/2*k2_g, u + 1/2*k2, r +1/2*h, E)
    k3_g = h*g(w + 1/2*k2_g, u + 1/2*k2, r +1/2*h, E)
    k4 = h*f(w +k3_g, u +k3, r+h, E)
    k4_g = h*g(w +k3_g, u +k3, r+h, E)
    u2 = u + 1/6*(k1+2*k2+2*k3+k4)
    w2 = w + 1/6*(k1_g+2*k2_g+2*k3_g+k4_g)
    return w2, u2

def Runge2(h,f,g,w,u,r, E): # Método de Runge-Kutta Orden 2 para ec. dif 2º ord.
    
    k1 = h*f(w,u,r,E)
    k1_g = h*g(w,u,r, E)
    k2 = h*f(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    k2_g = h*g(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    u2 = u + k2
    w2 = w + k2_g
    return w2, u2

def Leap(h,f,r,t,E,i, xapoyo): #Método Leapfrog para ec. dif 2º ord.
    if i == 0:
        x12 = r+h/2*f(r,t, E)
    elif i != 0:
        x12 = xapoyo
    x2 = r + h*f(x12, t+1/2*h, E)
    x32 = x12 + h*f(x2, t+h, E)
    xapoyo = x32
    return x2, xapoyo

def Euler(h,f,g,w,u,r, E): # Método de Euler para ec. dif 2º ord.
    u2 = u + h*f(w,u,r,E)
    w2 = w + h*g(w,u,r,E)
    return w2, u2

# Funciones de la ecuación diferencial. Ec. de Schrödinger.

def f1(w,u,r, E):
    dudr = w
    return dudr

def f2(w,u,r, E, tipo):
    if tipo == 1:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)))
    elif tipo == 2:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2)
    elif tipo == 3:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))
    elif tipo == 4:
         if r != 0:
             dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)*np.exp(-r/a)*4*a**2/(0.0483*(1-np.exp(-r/a)))**2)
         elif r == 0:
             dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2))
    elif tipo == 5:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2 -l*(l+1)/(0.0483*2*r**2))
    elif tipo ==6:
        if r != 0:
             dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2 -l*(l+1)*np.exp(-r/a)*4*a**2/(0.0483*(1-np.exp(-r/a)))**2)
        elif r == 0:
             dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2 -l*(l+1)/(0.0483*2))
    return dwdr

# Normalización.

def norm(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return array/tot


# Menús de selección de métodos y condiciones de contorno.

selec = int(input("Elija el método para la resolución de la ecuación de Schrödinger:\n 1. Runge-Kutta 4º orden. \n\
 2. Runge-Kutta 2º orden. \n 3. Leapfrog. \n 4. Euler. \n"))
        
(n, l) = (int(input("Número cuántico principal: \n")), int(input("Número cuántico azimutal: \n ")))

if l == 0:
    tipo = int(input("¿Cuál versión del potencial de Woods-Saxon?\n 1. Clásico \n 2. Modificado. \n"))

if l != 0:
    app = int(input("¿Realizar aproximación o tomar r cerca de 0? \n 1. Aproximación \n 2. Cerca de 0. \n"))
    if app == 1:
        tipo = int(input("¿Cuál versión del potencial de Woods-Saxon?\n 4. Clásico \n 6. Modificado. \n "))
    elif app == 2:
        tipo = int(input("¿Cuál versión del potencial de Woods-Saxon?\n 3. Clásico \n 5. Modificado. \n "))
    
if tipo == 3 or tipo == 5:
    x = np.linspace(1e-5, R + 15, 1000)      # Array de posiciones.

if tipo == 5 or tipo == 6 or tipo == 2:
    W0 = float(input("Introducir profundidad del pozo, W0 (float): \n "))
    
met = int(input("¿Cuál método de búsqueda de autovalores? \n 1. Bipartición. \n 2. Newton. \n"))

if met == 1:
    Nmax = 100
    prec = 1e-15
    if (n, l) == (0,0):
        Emax = -30
        Emin = -40
    elif (n, l) == (1, 0):
        Emax = -15
        Emin = -20
    elif (n, l) == (2, 0):
        Emax = -0
        Emin = -2  
    Rn = 15   
    En = (Emax + Emin)/2
    
if met == 2:
    Nmax = 100
    prec = 1e-15
    if (n, l) == (0,0):
        En = -38.2
        Eap = -38.2   
        dE0 = 1
        deltaE = 1e-4



