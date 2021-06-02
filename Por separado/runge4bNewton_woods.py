import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation


"""Implementación del algoritmo de Runge-Kutta de cuarto orden para
    ecuaciones diferenciales de segundo orden."""
    
A = 56                                     # Número másico. 
r0 = 1.285                                 # Constante en m.
a = 0.65                                   # Consante en m.
V0 = 47.78                                 # Potencial subcero en MeV.
En = -38.5
Eap = -38.5                                # Energía MeV.
R = r0*A**(1/3)                            # Constante radial corteza.
cte = 0.0483
Nmax = 500
prec = 1e-15
l = 0
n = 0
dE0 = 1
deltaE = 1e-6

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
               
def norm(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return array/tot
dx = 0.03
x = np.arange(0, R + 5, dx)      # Array de posiciones.

c = 0                                      # Contador

while dE0 > prec:
    c += 1
    if c == Nmax: 
        break
    u0 = 0                                     # Condición inicial para r*R(r)
    w0 = 1e-5                                 # Condición inicial para du/dr
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
                dE = -urunge4[-2]/((urunge4[-1] -urunge4[-2])/deltaE)
                print("a", dE)
                En = Eap - dE
                break
            elif urunge4[i+1] < 0:
                dE = -urunge4[-2]/((urunge4[-1] -urunge4[-2])/deltaE)
                print(dE)
                En = Eap + dE
                break
    dE0 = abs(Eap-En)
    Eap = En
print(En)

plt.title("U(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("U(r)")
ur4n = norm(urunge4, dx)
plt.plot(x,ur4n, color ="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
#plt.xlim(0,max(x))
#plt.ylim(-max(ur4n) - max(ur4n)/10, max(ur4n) + max(ur4n)/5)

fig2 = plt.figure()
plt.title("R(r) frente a la coordenada radial")
plt.xlabel("r (fm)")
plt.ylabel("R(r)")

rr = np.zeros(len(urunge4))
rr[1:] = ur4n[1:]/x[1:]
rrn = norm(rr, dx)
plt.plot(x[1:], rrn[1:], color="k", label="n = {}". format(n))
plt.legend(loc="best", frameon = False)
#plt.xlim(0,max(x))
#plt.ylim(min(rrn) + min(rrn)/10, max(rrn) + max(rrn)/5)

#np.savetxt("U_Runge4Newton_n1_18_57.txt", ur4n)
#np.savetxt("R_Runge4Newton_n2_0_35.txt", rrn)
