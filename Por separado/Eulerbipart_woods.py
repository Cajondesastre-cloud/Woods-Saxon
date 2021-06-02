import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation
from time import time

tiempoini = time()      # Inicialización del contador.


"""Implementación del algoritmo de Euler para
    ecuaciones diferenciales de segundo orden."""
    
A = 56                                     # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial subcero en MeV.
R = r0*A**(1/3)                            # Constante radial corteza.
Emax = -30
Emin = -40
En = (Emax + Emin)/2                       # Energía MeV.
cte = 0.0483
dE0 = Emax-Emin
Nmax = 100
prec = 1e-17
l = 0
n = 0

def Euler(h,f,g,w,u,r, E): # Método de Runge-Kutta Orden 4 para ec. dif 2º ord.

    u2 = u + h*f1(w,u,r,E)
    w2 = w + h*f2(w,u,r,E)
    return w2, u2

def f1(w,u,r, E):
    dudr = w
    return dudr

def f2(w,u,r, E):
    dwdt=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))
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
    w0 = 1e-5                              # Condición inicial para du/dr
    ueuler = []
    weuler = []
    ueuler.append(u0)
    weuler.append(w0)
    for i in range(len(x)-1):
        sol = Euler(dx,f1, f2,weuler[i], ueuler[i], x[i], En)
        ueuler.append(sol[1])
        weuler.append(sol[0])
        if i == len(x)-2:
            if ueuler[-1] > 0:
                Emin = En
                En = Emin + (Emax-Emin)/2
                break
            elif ueuler[-1] < 0:
                Emax = En
                En = Emax - (Emax-Emin)/2
                break
    dE0 = abs(Emax-Emin)
tiempo = time()-tiempoini # Tiempo final. 
print("Se ha tardado {} segundos.".format(tiempo))     
print(En, c)

fig1 = plt.figure()
plt.title("U(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("U(r)")
ur2n = norm(ueuler, dx)
plt.plot(x, ur2n, color ="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
plt.xlim(0,max(x))
plt.ylim(min(ur2n) +min(ur2n)/10, max(ur2n) + max(ur2n)/10)

fig2 = plt.figure()
plt.title("R(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("R(r)")
rr = ueuler/x
rrn = norm(rr[1:], dx)
plt.plot(x[1:], rrn, color="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
plt.xlim(0,max(x))
plt.ylim(min(rrn) +min(rrn)/10, max(rrn) + max(rrn)/10)

#np.savetxt("U_Eulerbipart_n2.txt", ur2n)
#np.savetxt("R_Eulerbipart_n2.txt", rrn)
