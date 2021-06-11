# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:28:31 2020

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as mcolors
import math as mt
import cmath as cm
import sympy as sp
from sympy import N
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

#Parámetros
#Números cuánticos
n = 2
l = 2
m = 0
inte = 0.12 #diferencial de x
#Variables
phi = np.linspace(0, 2*np.pi, 100)
theta = np.linspace(0, np.pi, 100)

#Parte ángular de la solución.

###########Parte Angular

############################################################
############################################################
############################################################
###########Parte Polar


fpol = np.exp(0. + (phi*m*1j))
impol=fpol.imag
repol=fpol.real
abspol=abs(fpol)

    #Representación en coordenadas polares.
    #Parte imaginaria

ax = plt.subplot(221,projection='polar')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

    #Gama de colores.
Escalapi = (impol - impol.min()) / impol.ptp()
colpi = plt.cm.coolwarm(Escalapi)
    #Representamos.
plt.scatter(phi,np.abs(impol), c=colpi)
ax.grid(True)
ax.set_title("Solución eje Polar (Im)", va='baseline',pad=20)


    #Parte real.
ax = plt.subplot(222,projection='polar')

    #Escalamos la gama de color.
Escalapr = (repol - repol.min()) / repol.ptp()
colre = plt.cm.coolwarm(Escalapr)
    #Representamos.
plt.scatter(phi,abs(repol), c=colre)
ax.grid(True)
ax.set_title("Solución eje Polar (Re)", va='baseline',pad=20)

    #Probabilidad
ax = plt.subplot(223,projection='polar')

    #Gama de colores.
Escalabs = (abspol - abspol.min()) / abspol.ptp()
colabs = plt.cm.coolwarm(Escalabs)
    #Representamos.
plt.scatter(phi,np.abs(abspol), c=colabs)
ax.grid(True)
ax.set_title("Solución eje Polar (Prob)", va='baseline',pad=20)

############################################################
############################################################
############################################################
###########Parte Azimutal

    #Cálculo de la derivada n-ésima por recursividad.

def dern(f,i):
    def der(x,h =inte):
        return (f(x+h) - f(x-h))/(2*h)
    if(i==0):
        return f
    if(i==1):
        return der
    else:
        return dern(der,i-1)
    
    #Cálculo de la función asociada de Legendre.

def Legendre(x,l):
    def pol(x):
        p = (x**2-1)**l
        return p
    dr = dern(pol, l)
    polLeg = 1/(2**l*mt.factorial(l))*dr(x)
    return polLeg

def LegAso(x,l,m):
    def polas(x):
        pa = Legendre(x,l)
        return pa
    dr2 = dern(polas, abs(m))
    polLegas = (1-x**2)**(abs(m)/2)*dr2(x)
    return polLegas

Fleg = LegAso(np.cos(theta), l, m)
Az = abs(Fleg)

    #Representación en coordenadas polares.
ax = plt.subplot(224, projection = 'polar')

    #Gama de colores.
EscalaAz = (Fleg-Fleg.min())/Fleg.ptp()
colaz = plt.cm.coolwarm(EscalaAz)
    #Representamos.
plt.scatter(theta, abs(Az), c = colaz)
ax.grid(True)
ax.set_title('Solución eje Azimutal', va = 'baseline',pad=20)
plt.show()

#########################################
#########################################
#########################################
###########Armónicos Esféricos.

    #Normalización de los armónicos.
    
def norm(l, m):
    n1 = (-1)**m
    n21 = (2*l+1)*mt.factorial(l-m)
    n22 = (4*np.pi*mt.factorial(l+m))
    n2 = np.sqrt(n21/n22)
    return n1*n2

def arm(x, y, l, m):
    normalizacion = norm(l,m)
    fpol = np.exp(0. + (x*m*1j))
    FAz = LegAso(np.cos(y),l,m)
    Y = normalizacion*fpol*FAz
    return Y

    #Variables y grids.
theta2 = np.linspace(0,np.pi,200)
phi2 =  np.linspace(0, 2*np.pi, 40)
TH, PH = np.meshgrid(theta2, phi2)

    #Coordenadas esférficas a cartesianas.
R = np.real(arm(PH, TH, l, m))**2
X = R*np.sin(TH)*np.cos(PH)
Y = R*np.sin(TH)*np.sin(PH)
Z = R*np.cos(TH)

    #Representación.
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('coolwarm')
norm2 = mcolors.Normalize(vmin=R.min(), vmax=R.max())
plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap(norm2(R)), linewidth=0, antialiased=False, alpha=.4)
ax.set_title("Orbital 1 1 0", va='bottom')
ax.set_xlabel(r"$Rsin(\theta)cos(\phi)$")
ax.set_ylabel(r"$Rsin(\theta)sin(\phi)$")
ax.set_zlabel(r"$Rcos(\theta)$")
plt.show()
plt.savefig("3D_1_2.png")
###########Parte Radial

"""Implementación del algoritmo de Runge-Kutta de cuarto orden para
    ecuaciones diferenciales de segundo orden."""
    
A = 56                                    # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial subcero en meV.
R = r0*A**(1/3)                            # Constante radial corteza.
Emax = -0
Emin = -10
En = (Emax+Emin)/2                         # Energía meV.
cte = 0.0483
dE0 = Emax-Emin
Nmax = 1000
prec = 1e-10
l = 2

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

def norma(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return array/tot

def f1(w,u,r, E):
    dudr = w
    return dudr

def f2_l(w, u, r, E):
    dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))
    return dwdr

r = np.linspace(1e-10, R+15, 1001)        # Array de posiciones.
dx = r[1]-r[0]
c = 0                                      # Contador  
while dE0 > prec:
    c += 1
    if c == Nmax: 
        break
    u0 = 0                                 # Condición inicial para r*R(r)
    w0 = 1e-5*(-1)**(n +1)                         # Condición inicial para du/dr
    urunge4 = []
    wrunge4 = []
    urunge4.append(u0)
    wrunge4.append(w0)
    for i in range(len(r)-1):
        sol = Runge4(dx,f1, f2_l ,wrunge4[i], urunge4[i], r[i], En)
        urunge4.append(sol[1])
        wrunge4.append(sol[0])
        if i == len(r)-2:
            if urunge4[-1] > 0:
                Emin = En
                En = Emin + (Emax-Emin)/2
                break
            elif urunge4[i+1] < 0:
                Emax = En
                En = Emax - (Emax-Emin)/2
                break
    dE0 = abs(Emax-Emin)

#Variables y ejecución.
ur4n = norma(urunge4, dx)
rr = ur4n/r
rrn = norma(rr, dx)
frad = rrn[1:]
    #Gama de colores.
Escalarad = (frad-frad.min())/frad.ptp()
colorad = plt.cm.coolwarm(Escalarad)

    #Representación.
plt.figure()
ax = plt.subplot(111)
plt.scatter(r[1:], frad, c=colorad)
ax.grid(True)
ax.set_title("Solución de la parte radial", va='bottom')
plt.show()

###########Solución Ec. Schrödinger

#Variables

r = np.linspace(0, 10*n, 1001)
th = np.linspace(0, 2.1*np.pi, 1001)
R, TH = np.meshgrid(r, th)

VAL = (np.abs(rr*arm(0,TH,l,m)))
X = R*np.cos(TH)
Y = R*np.sin(TH)

#Gama de colores
colorinterpolation = 50
colourMap = plt.cm.inferno

#Representación
fig = plt.figure(dpi = 400)
ax = fig.add_subplot(111)
plt.contourf(Y,X,VAL, colorinterpolation, cmap = colourMap)
plt.title('Tomografía de distrbución de probabilidad')
plt.show()
plt.savefig("Tom_1_2.png")




