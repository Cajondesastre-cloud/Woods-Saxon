import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation
from time import time

tiempoini = time()      # Inicialización del contador.

"""Implementación del algoritmo de Runge-Kutta de cuarto orden para
    ecuaciones diferenciales de segundo orden."""
    
A = 56                                     # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial subcero en meV.
R = r0*A**(1/3)                            # Constante radial corteza.
Emax = -30
Emin = -40
En = (Emax+Emin)/2                         # Energía meV.
cte = 0.0483
dE0 = Emax-Emin
Nmax = 1000
prec = 1e-10
l = 0
n = 0
tol = 1e-12

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

"""
def adapt_size(h, f, g, w, u, r, E, tol):
    w_p, u_p = Runge4(h, f, g, w, u, r, E)
    w_p2, u_p2 = Runge4(h, f, g, w_p, u_p, r + h, E)
    w_p3, u_p3 = Runge4(2*h, f, g, w, u, r, E)
    e_u = 1/30*(u_p2 - u_p3)
    e_w = 1/30*(w_p2-w_p3)
    hnew = h
    if e_u != 0 or e_w != 0:
        hnew = h*(30*h*tol/(np.sqrt(e_u**2 + e_w**2)))
    
    if hnew < h:
        wn, un = Runge4(hnew, f, g, w, u, r, E)
        return wn, un, hnew
    else:
        h = hnew
        return w_p, u_p, hnew
"""    
def f1(w,u,r, E):
    dudr = w
    return dudr

def f2_mod(w,u,r,E):
    W0 = -100
    dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2)
    return dwdr

def f2(w,u,r, E):
    dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)))
    return dwdr

def f2_l(w, u, r, E):
    dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))
    return dwdr
        
def f2_l_alt(w,u,r, E):
    if r != 0:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)*np.exp(-r/a)*4*a**2/(0.0483*(1-np.exp(-r/a)))**2)
    elif r == 0:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2))
    return dwdr
        
def norm(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return array/tot

def norm2(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h[i]
    return array/tot


def dens(array, h):
    tot = 0
    for i in range(len(array)):
        if i > 150:
            break
        tot += abs(array[i])**2*h
    return tot


def dens2(array, h):
    tot = 0
    for i in range(len(array)):
        if i > 150:
            break
        tot += abs(array[i])**2*h[i]
    return tot


c = 0                                      # Contador 

while dE0 > prec:
   
    c += 1
    if c == Nmax: break;
        
    u0 = 0                                     # Condición inicial para r*R(r)
    w0 = 1e-5*(-1)**(n)                        # Condición inicial para du/dr
    urunge4 = []
    wrunge4 = []
    urunge4.append(u0)
    wrunge4.append(w0)
    x = []
    x.append(0)
    x_act = 0
    h = 0.013
    i = 0
    h_list = []
    h_list.append(h)
    while x_act < R+15:
        w_p, u_p = Runge4(h,f1, f2 ,wrunge4[i], urunge4[i], x_act, En)
        w_p2, u_p2 = Runge4(h, f1, f2, w_p, u_p, x_act + h, En)
        w_p3, u_p3 = Runge4(2*h, f1, f2, wrunge4[i], urunge4[i], x_act, En)
        e_u = 1/30*(u_p2 - u_p3)
        e_w = 1/30*(w_p2 - w_p3)
       
        hnew = h*(30*h*tol/(np.sqrt(e_u**2 + e_w**2)))**(1/4)
        
        if hnew < h:
            w_p, u_p = Runge4(hnew, f1, f2, wrunge4[i], urunge4[i], x_act, En)
            h = hnew
        else:
            #h = h + tol*1e6
            k = h*1.005
            if k > 0.05:
                h = hnew
            else:
                h = k
        urunge4.append(u_p)
        wrunge4.append(w_p)
        x_act += h
        x.append(x_act)
        h_list.append(h)
        if x_act > R+ 15 - h:
            if urunge4[-1] > 0:
                Emin = En
                En = Emin + (Emax-Emin)/2
                break
            elif urunge4[-1] < 0:
                Emax = En
                En = Emax - (Emax-Emin)/2
                break
        i += 1
    dE0 = abs(Emax-Emin)
    

tiempo = time()-tiempoini # Tiempo final. 
print("Se ha tardado {} segundos.".format(tiempo))    
print(En, c)
plt.title("U(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("U(r)")
ur4n = norm2(urunge4, h_list)
plt.plot(x,ur4n, color ="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)


fig2 = plt.figure()
plt.title("R(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("R(r)")
rr = ur4n[1:]/x[1:]
rrn = norm2(rr, h_list)
plt.plot(x[1:], rrn[:], color="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)


rarray = rrn[1:].copy()
nuc_dens = dens2(rarray, h_list)
fac = 1.66054e-27                          # 1 uma tantos kilogramos.
fac2 = (1e-15)**3                          # 1 fm^3 tantos m^3
print(nuc_dens)
print(nuc_dens/fac2*fac)

#np.savetxt("Adapt.txt", rrn)
#np.savetxt("x.txt", x)
#np.savetxt("h.txt", h_list)