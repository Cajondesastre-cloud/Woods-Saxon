# Código del programa empleado para obtener resultados en el TFG del grado
# en Física de la Universidad de Alicante: Estudio numérico del potencial
# de Woods-Saxon. Elaborado por David Montiel López (dml22@alu.ua.es).

import numpy as np
import matplotlib.pyplot as plt
import argparse

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
metnam = 0                                 # Contendrá el nombre del método.

# Normalización.

def norm(array, h):
    tot = 0
    for i in range(len(array)):
        tot += abs(array[i])*h
    return array/tot

##############################################################################
##############################################################################
##############################################################################
#######################################################
#######################################################
#############################################
#############################################

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

def Leap(h,f,r,t,E,i, xapoyo):    #Método Leapfrog para ec. dif 2º ord.
    if i == 0:
        x12 = r+h/2*f(r,t, E)
    elif i != 0:
        x12 = xapoyo
    x2 = r + h*f(x12, t+1/2*h, E)
    x32 = x12 + h*f(x2, t+h, E)
    xapoyo = x32
    return x2, xapoyo

def Euler(h,f,g,w,u,r, E, tipo):        # Método de Euler para ec. dif 2º ord.
    u2 = u + h*f(w,u,r,E)
    w2 = w + h*g(w,u,r,E, tipo)
    return w2, u2

#############################################
#############################################
#######################################################
#######################################################
##############################################################################
##############################################################################
##############################################################################

# Menú, argparser.

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", help="Differential equation solving method:\n \
1) Runge4, 2) Runge2, 3) Leap or 4) Euler.", type=int)
parser.add_argument("-g", "--gen", help="Use generalized Woods-Saxon potential.",
action= "store_true")
parser.add_argument("-n", "--ncuant", help="Principal quantum number.", type=int)
parser.add_argument("-l", "--lcuant", help="Azimutal quantum number.", type=int)
parser.add_argument("-w", "--potW", help="Potential depth (gen Woods-Saxon).", type=float)
parser.add_argument("-r", "--rep", help="Activate result plotting.",
action= "store_true")
parser.add_argument( "-s", "--save", nargs=2, help="Activate saving results. Needs two strings using space as separator (two figures).")
args = parser.parse_args()

##############################################################################
##############################################################################
##############################################################################
#######################################################
#######################################################
#############################################
#############################################

# Funciones de la ecuación diferencial. Ec. de Schrödinger.

def f1(w,u,r, E):
    dudr = w
    return dudr

def f2(w,u,r, E):
    if args.lcuant == 0 and not args.gen:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)))
    elif args.lcuant == 0 and args.gen:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) 
                        +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2)
    elif args.lcuant != 0 and not args.gen:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))

    elif args.lcuant != 0 and args.gen:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/
                        (1 + np.exp((r-R)/a))**2 -l*(l+1)/(0.0483*2*r**2))
    return dwdr

def f(r, t, E):
    dwdt=np.array([r[1], -r[0]*0.0483*(E + V0/(1+np.exp((t-R)/a)))])
    return dwdt

#############################################
#############################################
#######################################################
#######################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
#######################################################
#######################################################
#############################################
#############################################

if args.method == 1:
    func = Runge4
    metnam = "Runge-Kutta 4º orden"
elif args.method == 2:
    func = Runge2
    metnam = "Runge-Kutta 2º orden"
elif args.method == 3:
    func = Leap
    metnam = "Leapfrog"
elif args.method == 4:
    func = Euler
    metnam = "Euler"
        
(n, l) = (args.ncuant, args.lcuant)
                
if args.gen:
    W0 = args.potW
            
if args.lcuant != 0:
    x = np.linspace(1e-5, R + Rn, 1000)      # Array de posiciones.
    dx = x[1] - x[0]                         # Diferencial de x.
            
Nmax = 100               # Número máximo de iteraciones.
prec = 1e-15             # Preción máxima requerida.
if (n, l) == (0, 0):      # Cotas de la energía para cada estado.
    Emax = -30
    Emin = -40
elif (n, l) == (1, 0):
    Emax = -15
    Emin = -20
elif (n, l) == (2, 0):
    Emax = -0
    Emin = -2  
elif (n, l) == (0,1):
    Emax = -30
    Emin = -35
elif (n, l) == (0,2):
    Emax = -20
    Emin = -30
elif (n, l) == (1,1):
    Emax = -10
    Emin = -15
elif (n, l) == (1,2):
    Emax = -1
    Emin = -7   
Rn = 15   
En = (Emax + Emin)/2
dE0 = abs(En)
            
if args.method != 3:
    c = 0                           # Contador
    while dE0 > prec:
        c += 1
        if c == Nmax: 
            break
        u0 = 0                      # Condición inicial para r*R(r)
        w0 = 1e-5*(-1)**(n)         # Condición inicial para du/dr
        ur = []
        wr = []
        ur.append(u0)
        wr.append(w0)
        for i in range(len(x)-1):
            sol = func(dx,f1, f2 ,wr[i], ur[i], x[i], En)
            ur.append(sol[1])
            wr.append(sol[0])
            if i == len(x)-2:
                if ur[-1] > 0:
                    Emin = En
                    En = Emin + (Emax-Emin)/2
                    break
                elif ur[-1] < 0:
                    Emax = En
                    En = Emax - (Emax-Emin)/2
                    break
        dE0 = abs(Emax-Emin)
                    
    print(En, "MeV")
                
else:
    c = 0                           # Contador
    while dE0 > prec:
        c += 1
        if c == Nmax: 
            break
        u0 = 0                       # Condición inicial para r*R(r)
        w0 = 1e-5*(-1)**(n)          # Condición inicial para du/dr
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
    
    print(En, "MeV")
    ur = []
    wr = []
    for i in oscleap:
        ur.append(i[0])
        wr.append(i[1])
        
if args.rep:
    fig1 =plt.figure()
    plt.title("U(r) frente a la coordenada radial: método de {}".format(metnam))
    plt.xlabel("r (fm)")
    plt.ylabel("U(r)")
    urn = norm(ur, dx)
    plt.plot(x,urn, color ="k", label="(n,l) = ({},{})". format(n, l))
    plt.legend(loc="best", frameon = False)
    plt.show()
            
    if args.save:
        plt.savefig((args.save[0]), dpi= 400)
                
    fig2 = plt.figure()
    plt.title("R(r) frente a la coordenada radial: método de {}".format(metnam))
    plt.xlabel("r (fm)")
    plt.ylabel("R(r)")
    rr = np.zeros(len(ur))
    rr[1:] = urn[1:]/x[1:]
    rrn = norm(rr, dx)
    plt.plot(x[1:], rrn[1:], color="k", label="(n,l) = ({},{})". format(n, l))
    plt.legend(loc="best", frameon = False)
    plt.show()
    if args.save:
        plt.savefig(args.save[1], dpi= 400)
                    
#############################################
#############################################
#######################################################
#######################################################
##############################################################################
##############################################################################
##############################################################################