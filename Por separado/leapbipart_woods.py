import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation
from time import time

tiempoini = time()      # Inicialización del contador.

"""Implementación del algoritmo de Leapfrog para
    ecuaciones diferenciales de segundo orden."""
    
A = 56                                     # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial subcero en MeV.
R = r0*A**(1/3)                            # Constante radial corteza.
Emax = -30
Emin = -40
cte = 0.0483
dE0 = Emax-Emin
Nmax = 100
prec = 1e-10
En = Emax/2 + Emin/2                       # Energía MeV.
l = 0
n = 0

def Leap(h,f,r,t,E,i, xapoyo): #Método Leapfrog.
    if i == 0:
        x12 = r+h/2*f(r,t, E)
    elif i != 0:
        x12 = xapoyo
    x2 = r + h*f(x12, t+1/2*h, E)
    x32 = x12 + h*f(x2, t+h, E)
    xapoyo = x32
    return x2, xapoyo

def f(r, t, E):
    dwdt=np.array([r[1], -r[0]*0.0483*(E + V0/(1+np.exp((t-R)/a)) -l*(l+1)/(0.0483*2*t**2))])
    return dwdt

def norm(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return (array)/tot

x = np.linspace(1e-5, R + 15, 1000)      # Array de posiciones.
dx = x[1]-x[0]

c = 0                                      # Contador

while dE0 > prec:
    c += 1
    if c == Nmax: 
        break
    u0 = 0                                  # Condición inicial para r*R(r)
    w0 = 1e-5                               # Condición inicial para du/dr
    oscleap=0
    oscleap = []
    oscleap.append([u0,w0])
    roscapoyo = [0,0]
    for i in range(len(x)-1):
        oscleap.append(Leap(dx,f,oscleap[i],x[i], En, i, roscapoyo)[0])
        roscapoyo = Leap(dx,f,oscleap[i], x[i],En, i, roscapoyo)[1]
        if i == len(x)-2:
            if oscleap[-1][0] > 0:
                Emin = En
                En = Emin + (Emax-Emin)/2
                break
            elif oscleap[-1][0] < 0:
                Emax = En
                En = Emax - (Emax-Emin)/2
                break
    dE0 = abs(Emax-Emin)
tiempo = time()-tiempoini # Tiempo final. 
print("Se ha tardado {} segundos.".format(tiempo))         
print(En, c)

xleap = []
yleap = []

for i in oscleap:
    xleap.append(i[0])
    yleap.append(i[1])


fig1 = plt.figure()
plt.title("U(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("U(r)")
urln = norm(xleap, dx)
plt.plot(x,urln, color ="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
plt.xlim(0,max(x))
#plt.ylim(-(max(urln) + max(urln)/10), max(urln) + max(urln)/10)

fig2 = plt.figure()
plt.title("R(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("R(r)")
rr = xleap/x
rrn = norm(rr[1:], dx)

plt.plot(x[1:], rrn, color="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
plt.xlim(0,max(x))
#plt.ylim(-(max(urln) + max(urln)/10), max(rrn) + max(rrn)/10)

#np.savetxt("U_Leapbipart_n2.txt", urln)
#np.savetxt("R_Leapbipart_n2.txt", rrn)
