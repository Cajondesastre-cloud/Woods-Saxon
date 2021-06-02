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

def dens(array, h):
    tot = 0
    for i in range(len(array)):
        if i > 150:
            break
        tot += abs(array[i])**2*h
    return tot

x = np.linspace(0, R+15, 1000)        # Array de posiciones.
dx = x[1]-x[0]
print(dx)
c = 0                                      # Contador
    
while dE0 > prec:
    c += 1
    if c == Nmax: 
        break
    u0 = 0                                 # Condición inicial para r*R(r)
    w0 = 1e-5*(-1)**n                          # Condición inicial para du/dr
    urunge4 = []
    wrunge4 = []
    urunge4.append(u0)
    wrunge4.append(w0)
    for i in range(len(x)-1):
        sol = Runge4(dx,f1, f2 ,wrunge4[i], urunge4[i], x[i], En)
        urunge4.append(sol[1])
        wrunge4.append(sol[0])
        if i == len(x)-2:
            if urunge4[-1] > 0:
                Emin = En
                En = Emin + (Emax-Emin)/2
                break
            elif urunge4[-1] < 0:
                Emax = En
                En = Emax - (Emax-Emin)/2
                break
    dE0 = abs(Emax-Emin)

tiempo = time()-tiempoini # Tiempo final. 
print("Se ha tardado {} segundos.".format(tiempo))    
print(En, c)

ur4n = norm(urunge4, dx)
rr = ur4n[1:]/x[1:]
rrn = norm(rr, dx)

# Animación.

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("r[fm]")
ax.set_ylabel("rR(r)")
ax.set_xlim(0,R+15)
ax.set_ylim(0, 0.4)
txt = fig.suptitle('')

def paso(i):
    ploty = rrn[i]
    r = x[i]
    func, = ax.plot(r, ploty, marker = '.', c = 'red', linestyle='None')
    txt.set_text('Paso = {}'.format(i))
    return txt

anim = animation.FuncAnimation(fig, paso, frames=(len(x)), interval=100,
                                repeat= False)

plt.show()

# Grabación.

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim.save('RK4_0_f.mp4', writer=writer)


#anim.save('Temperatura.gif', writer="imagemagick", fps=30)
