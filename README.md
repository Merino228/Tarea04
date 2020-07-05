# Tarea04
Tarea 4 Modelos Probabilisticos

En esta tarea se comenzoo por importar los datos provenientes del arcvhivo *bits10k.csv*. Dichos datos se almacenaron en una variable llamada *bits*. Luego se importaron las siguientes librerias:

    import numpy as np
    import pandas as pd
    from scipy.linalg import inv
    from scipy import stats
    from scipy import signal
    from scipy import integrate
    import matplotlib.pyplot as plt
    import csv

Primero se definen las variables de interes para la resolución del problema:  

Cantidad de bits

    N = lens(bits)
    
Frecuencia de la Portadora

    f = 5000
    
Periodo de cada simbolo

    T = 1/f

Numero de puntos para el muestreo por cada periodo 

    p = 50
    
Vector de los puntos de muestreo por cada periodo

    tp = np.linspace(0,T,p)
    
Forma de onda de la Portadora (Onda Sinusoidal) 

    sinus = np.sin(2*np.pi * f * tp)
    
Frecuencia de Muestreo

    fm = p/T
    

    

## Pregunta 1. Modulación BPSK
Para la modulación BPSK se comenzó por crear un vector "temporal" para la señal emitida (*Tx*) 

    t = np.linspace(0,N*T,N*p)
    
Luego se creó una variable en donde se va a almacenar la nueva señal


    senal = np.zeros(t.shape)
   
Finalmente para esa parte se creo la señal modulada con:

    for k, b in enumerate(bits):
        bitActual = bits[k]
        if bitActual == 1:
            senal[k * p:(k + 1) * p] = sinus
        else:
            senal[k * p:(k + 1) * p] = -sinus
 
Dicho segmento de código basicamente adiciona en la señal de almacenamiento, un *Seno* en la ubicación de cada 1 en el array *bits* y una señal *-Seno* en la ubicación de cada 0 en el mismo array. Generando asi a siguiente senal modulada:


## Pregunta 2. Potencia promedio de la señal modulada
Se calcula la potencia instantanea de la señal como:

        Pinst = senal**2

A partir de la cual se puede calcular la potencia promedio en Watts

        Ps = integrate.trapz(Pinst, t) / (N * T)
                
## Pregunta 3. Canal ruidoso AWGN con una relacion señal a ruido (*SNR*) desde -2 hasta 3dB
Primero establecemos el rango de posibles valores de SNR

        rango = range(-2,4) # Genera un rango de [-2,3]


## Pregunta 4.
## Pregunta 5.
## Pregunta 6. 
