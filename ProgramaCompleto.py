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
r0 = 1.285                                 # Constante en fm.
a = 0.65                                   # Consante en fm.
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

def Runge4(h,f,g,w,u,r, E, tipo): # Método de Runge-Kutta Orden 4 para ec. dif 2º ord.
    
    k1 = h*f(w,u,r,E)
    k1_g = h*g(w,u,r, E, tipo)
    k2 = h*f(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    k2_g = h*g(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E, tipo)
    k3 = h*f(w + 1/2*k2_g, u + 1/2*k2, r +1/2*h, E)
    k3_g = h*g(w + 1/2*k2_g, u + 1/2*k2, r +1/2*h, E, tipo)
    k4 = h*f(w +k3_g, u +k3, r+h, E)
    k4_g = h*g(w +k3_g, u +k3, r+h, E, tipo)
    u2 = u + 1/6*(k1+2*k2+2*k3+k4)
    w2 = w + 1/6*(k1_g+2*k2_g+2*k3_g+k4_g)
    return w2, u2

def Runge2(h,f,g,w,u,r, E, tipo): # Método de Runge-Kutta Orden 2 para ec. dif 2º ord.
    
    k1 = h*f(w,u,r,E)
    k1_g = h*g(w,u,r, E, tipo)
    k2 = h*f(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E)
    k2_g = h*g(w +1/2*k1_g, u +1/2*k1, r +1/2*h, E, tipo)
    u2 = u + k2
    w2 = w + k2_g
    return w2, u2

def Leap(h,f,r,t,E,i, xapoyo, tipo):    #Método Leapfrog para ec. dif 2º ord.
    if i == 0:
        x12 = r+h/2*f(r,t, E, tipo)
    elif i != 0:
        x12 = xapoyo
    x2 = r + h*f(x12, t+1/2*h, E, tipo)
    x32 = x12 + h*f(x2, t+h, E, tipo)
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

def f2(w,u,r, E, tipo):
    if tipo == 1:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)))
    elif tipo == 2:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) 
                        +W0*np.exp((r-R)/a)/(1 + np.exp((r-R)/a))**2)
    elif tipo == 3:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) -l*(l+1)/(0.0483*2*r**2))

    elif tipo == 4:
        dwdr=-u*0.0483*(E + V0/(1+np.exp((r-R)/a)) +W0*np.exp((r-R)/a)/
                        (1 + np.exp((r-R)/a))**2 -l*(l+1)/(0.0483*2*r**2))
 
    return dwdr

def f(r, t, E, tipo):
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

# Menús de selección de métodos y condiciones de contorno.

menu = input("¿Quiere seleccionar opciones de representación? \
(En caso negativo se ejecutará rápidamente) \n Sí (s) o No (n): \n")
             
if menu == "s":
    acabar = False                          # Booleano para repetición de menú.
    while acabar == False:
    
        selec = int(input("Elija el método para la resolución de la ecuación \
de Schrödinger:\n 1. Runge-Kutta 4º orden. \n\
 2. Runge-Kutta 2º orden. \n 3. Leapfrog. \n 4. Euler. \n"))
         
        if selec == 1:
            func = Runge4
            metnam = "Runge-Kutta 4º orden"
        elif selec == 2:
            func = Runge2
            metnam = "Runge-Kutta 2º orden"
        elif selec == 3:
            func = Leap
            metnam = "Leapfrog"
        elif selec == 4:
            func = Euler
            metnam = "Euler"
        
        (n, l) = (int(input("Número cuántico principal: \n")), 
                  int(input("Número cuántico azimutal: \n")))
                
        if l == 0:
            tipo = int(input("¿Cuál versión del potencial de Woods-Saxon? \
\n 1. Clásico \n 2. Generalizado. \n"))
        
        if l != 0:
            tipo = int(input("¿Cuál versión del potencial de Woods-Saxon? \
\n 3. Clásico \n 4. Generalizado. \n "))
            
        if tipo == 3 or tipo == 4:
            x = np.linspace(1e-5, R + Rn, 1000)      # Array de posiciones.
            dx = x[1] - x[0]                         # Diferencial de x.
        
        if tipo == 4 or tipo == 2:
            W0 = float(input("Introducir profundidad del pozo, W0 (float): \n "))
            
        met = int(input("¿Cuál método de búsqueda de autovalores? \
\n 1. Bipartición. \n"))
        
        if met == 1:                 # Resolviendo por bipartición.
            Nmax = 100               # Número máximo de iteraciones.
            prec = 1e-15             # Preción máxima requerida.
            if (n, l) == (0,0):      # Cotas de la energía para cada estado.
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
            
            if selec != 3:
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
                        sol = func(dx,f1, f2 ,wr[i], ur[i], x[i], En, tipo)
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
                
            if selec == 3:
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
                        oscleap.append(Leap(dx,f,oscleap[i],x[i], En, i, roscapoyo, tipo)[0])
                        roscapoyo = Leap(dx,f,oscleap[i], x[i],En, i, roscapoyo, tipo)[1]
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
              
        rep = input("¿Quiere representar los resultados (s/n)?\n")
        
        if rep == "s":
            
            sav = input("¿Quiere guardar los resultados (s/n)?\n")
            
            plt.title("U(r) frente a la coordenada radial: método de {}".format(metnam))
            plt.xlabel("r (fm)")
            plt.ylabel("U(r)")
            urn = norm(ur, dx)
            plt.plot(x,urn, color ="k", label="(n,l) = ({},{})". format(n, l))
            plt.legend(loc="best", frameon = False)
            plt.show()
            
            if sav == "s":
                plt.savefig(input("Introduzca el nombre de la figura 1: "), dpi= 400)
                
            fig2 = plt.figure()
            plt.title("R(r) frente a la coordenada radial: método de {}".format(metnam))
            plt.xlabel("r (fm)")
            plt.ylabel("R(r)")
            rr = np.zeros(len(ur))
            rr[1:] = urn[1:]/x[1:]
            rrn = norm(rr, dx)
            plt.plot(x[1:], rrn[1:], color="k", label="(n,l) = ({},{})". format(n, l))
            plt.legend(loc="best", frameon = False)
                 
            if sav == "s":
                plt.savefig(input("Introduzca el nombre de la figura 2: "), dpi= 400)
        
        fin = input("¿Quiere seguir haciendo simulaciones (s/n)?\n")
        
        if fin == "n":
            acabar = True
            
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

# Ejecución rápida, se le pasa un array con las opciones.

elif menu == "n":
    acabar = True
    print("El formato del archivo de datos a cargar será como el que se incluye como ejemplo.")
    print(" Array ejemplo: 1 0 0 1 1 1 0 50 usando espacios como separador")
    print(" 1º = método (1-4), 2º y 3º = (n,l), 4º potencial clásico o generalizado (1-2),\
 5º = prueba y error (1), 6º representar? (0 = No, 1 = Sí), 7º Guardar (0-1) y 8º W0 en caso\
 de que el potencial sea generalizado.")
    conf = np.loadtxt(input(" Nombre del archivo a cargar (incluyendo extensión, .txt o .dat preferentemente): \n"))
    selec = conf[0]
    (n, l) = (conf[1], conf[2])
    tipo = conf[3]
    met = conf[4]
    rep = "n"
    if conf[5] == 1:
        rep = "s"
    sav = "n"
    if conf[6] == 1:
        sav = "s"
    W0 = conf[7]
    
    if selec == 1:
        func = Runge4
        metnam = "Runge-Kutta 4º orden"
    elif selec == 2:
        func = Runge2
        metnam = "Runge-Kutta 2º orden"
    elif selec == 3:
        func = Leap
        metnam = "Leapfrog"
    elif selec == 4:
        func = Euler
        metnam = "Euler"
        
    if l != 0:
        if tipo == 1:
            tipo = 3
        if tipo == 2:
            tipo == 4
            
    if tipo == 3 or tipo == 4:
        x = np.linspace(1e-5, R + Rn, 1000)      # Array de posiciones.
        dx = x[1] - x[0]                         # Diferencial de x.
        
     
    if met == 1:                 # Resolviendo por bipartición.
        Nmax = 100               # Número máximo de iteraciones.
        prec = 1e-15             # Preción máxima requerida.
        if (n, l) == (0,0):      # Cotas de la energía para cada estado.
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
            
        if selec != 3:
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
                    sol = func(dx,f1, f2 ,wr[i], ur[i], x[i], En, tipo)
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
                
        if selec == 3:
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
                    oscleap.append(Leap(dx,f,oscleap[i],x[i], En, i, roscapoyo, tipo)[0])
                    roscapoyo = Leap(dx,f,oscleap[i], x[i],En, i, roscapoyo, tipo)[1]
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
                 
        if rep == "s":
            
            plt.title("U(r) frente a la coordenada radial: método de {}".format(metnam))
            plt.xlabel("r (fm)")
            plt.ylabel("U(r)")
            urn = norm(ur, dx)
            plt.plot(x,urn, color ="k", label="(n,l) = ({},{})". format(n, l))
            plt.legend(loc="best", frameon = False)
            plt.show()
            
            if sav == "s":
                plt.savefig(input("Introduzca el nombre de la figura 1: "), dpi= 400)
                
            fig2 = plt.figure()
            plt.title("R(r) frente a la coordenada radial: método de {}".format(metnam))
            plt.xlabel("r (fm)")
            plt.ylabel("R(r)")
            rr = np.zeros(len(ur))
            rr[1:] = urn[1:]/x[1:]
            rrn = norm(rr, dx)
            plt.plot(x[1:], rrn[1:], color="k", label="(n,l) = ({},{})". format(n, l))
            plt.legend(loc="best", frameon = False)
                 
            if sav == "s":
                plt.savefig(input("Introduzca el nombre de la figura 2: "), dpi= 400)

##############################################################################
##############################################################################
############################################################
############################################################
#########################################
#########################################
