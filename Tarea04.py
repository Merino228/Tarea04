
'''
Tarea 04
Adrian Merino Leon - Paez
B74762
'''

import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy import stats
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt
import csv

# Extraemos los Datos del .CSV sin utilizar Pandas
with open('bits10k.csv') as f:
    bits = [int(s) for line in f.readlines() for s in line[:-1].split(',')]

pb = 7
primeros = bits[0:pb]
print('Primeros 7 bits ',primeros)
'''
Pregunta 1. Modulacion BPSK
'''
# Catnidad de Bits
N = len(bits)

# Frecuecia de la Portadora
f = 5000

# Periodo de cada simbolo
T = 1/f

# Numero de puntos de muestreo por periodo
p = 50

# Puntos de muestreo para cada período
tp = np.linspace(0, T, p)

# Creación de la forma de onda de la portadora
sinus = np.sin(2*np.pi * f * tp)


# Senal Sinusoidal
plt.figure()
plt.plot(tp, sinus)
plt.show()

'''
plt.figure()
plt.plot(tp,-sinus)
plt.show()
'''

# Frecuencia de muestreo
fm = p/T

# Tiempo para toda la señal Tx
t = np.linspace(0, N*T, N*p)

# Inicializar el vector de la señal
senal = np.zeros(t.shape)
lista=list(enumerate(bits))

for k, b in enumerate(bits):
    bitActual = bits[k]
    if bitActual == 1:
        senal[k * p:(k + 1) * p] = sinus
    else:
        senal[k * p:(k + 1) * p] = -sinus

plt.figure()
plt.plot(senal[0:pb*p])
plt.title('Senal Modulada')
plt.show()

'''
Pregunta 2.
'''

# Potencia instantánea
Pinst = senal**2

# Potencia promedio (W)
Ps = integrate.trapz(Pinst, t) / (N * T)

print('potencia Promedio',Ps)
'''
Pregunta 3.
'''

# Relacion  senal a ruido SNR
rango = range(-2,4)

for SNR in rango:
    print('SNR: ',SNR)
    # Potencia del ruido para SNR
    Pn = Ps /(10**(SNR/10))
    # Desviacion estandar del ruido
    sigma = np.sqrt(Pn)
    # Ruido con distribucion normal
    ruido = np.random.normal(0, sigma, senal.shape)
    # Senal con ruido
    Rx = senal + ruido


    # Graficamos Los primeros Bits recibidos
    plt.figure()
    plt.title ('Bits Recibidos Rx')
    plt.plot(Rx[0:pb*p])
    plt.show()

    # Pregunta 4.

    frec, Pot_antes = signal.welch(senal,fm)
    plt.semilogy(frec, Pot_antes)
    plt.title('Densidad espectral de potencia Antes del canal ruidoso')
    plt.show()

    frec, Pot_despues = signal.welch(Rx,fm)
    plt.semilogy(frec,Pot_despues)
    plt.title('Densidad espectral de potencia Despues del canal ruidoso')
    plt.show()


    # Pregunta 5.
# "Estandar" de energia de la onda original
    Es = np.sum(sinus**2)

# Vector para bits Recibidos
    bitsRx = np.zeros(N)

# Decodificación de la señal por detección de energía
    for k, b in enumerate(bits):
        # Producto interno de dos funciones
        Ep = np.sum(Rx[k*p:(k+1)*p] * sinus)
        if Ep > Es/2:
            bitsRx[k] = 1
        else:
            bitsRx[k] = 0
    print('Primeros Bits Recibidos: ',bitsRx[0:pb])
    # Cantidad de errores entre los enviados y los recibidos
    err = np.sum(np.abs(bits - bitsRx))
    print('error ', err)
    # Tasa de Error
    BER = err/N
    print('BER ',BER)


# Pregunta 6.

#Graficamos BER vs SNR

BERsim = [0.0022,0.0005,0.0,0.0,0.0,0]
SNRsim = [-2,-1,0,1,2,3]
plt.figure()
plt.plot(SNRsim, BERsim)
plt.title('BER vs SNR')
plt.ylabel('BER')
plt.xlabel('SNR')
plt.show()