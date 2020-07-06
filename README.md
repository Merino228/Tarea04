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

Cantidad de bits 10 000

    N = lens(bits)
    
Frecuencia de la Portadora 5000 Hz

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
 
Dicho segmento de código basicamente adiciona en la señal de almacenamiento, un *Seno* en la ubicación de cada 1 en el array *bits* y una señal *-Seno* en la ubicación de cada 0 en el mismo array. Generando asi a siguiente senal modulada para los primeros 7 bits:


## Pregunta 2. Potencia promedio de la señal modulada
Se calcula la potencia instantanea de la señal como:

        Pinst = senal**2

A partir de la cual se puede calcular la potencia promedio en Watts

        Ps = integrate.trapz(Pinst, t) / (N * T)
                
## Pregunta 3. Canal ruidoso AWGN con una relacion señal a ruido (*SNR*) desde -2 hasta 3dB
Primero establecemos el rango de posibles valores de SNR

        rango = range(-2,4) # Genera un rango de [-2,3]

Para dicho rango se calculó la potencia del ruido, con el cual se puede calcular su desviacion estandar con el cual se genera un ruido aditivo blanco gaussiano (AWGN) 

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

Al Graficar dicha señal con ruido para los primeros 7 bits obtenemos:

* SNR = -2

* SNR = -1

* SNR = 0

* SNR = 1

* SNR = 2

* SNR = 3

## Pregunta 4. Densidad espectral de potencia de la señal con el método de Welch (SciPy), antes y después del canal ruidoso.

La densidad espectral de potencia de la señal antes del canal ruidoso se calculó de la siguiente manera:

        frec, Pot_antes = signal.welch(senal,fm)
        plt.semilogy(frec, Pot_antes)
        plt.title('Densidad espectral de potencia Antes del canal ruidoso')
        plt.show()

Del mismo modo se calculó la densidad espectral de potencia de la señal despues del canal ruidoso:

        frec, Pot_despues = signal.welch(Rx,fm)
        plt.semilogy(frec,Pot_despues)
        plt.title('Densidad espectral de potencia Despues del canal ruidoso')
        plt.show()

Por lo tanto obtenemos las siguientes figuras para cada SNR

* SNR = -2

* SNR = -1

* SNR = 0

* SNR = 1

* SNR = 2

* SNR = 3

## Pregunta 5. Demodular y decodificar la señal y hacer un conteo de la tasa de error de bits (BER) para cada nivel SNR
Primero se calcula cual deberia ser un "estandar" de la energia generada por la onda original

        # "Estandar" de energia de la onda original
            Es = np.sum(sinus**2)

Luego, creamos una variable en donde se guardó los bits decodificados:

        # Vector para bits Recibidos
            bitsRx = np.zeros(N)

Finalmente, creamos el algoritmo de decodificación comparando la potencia de la señal recibida con la potencia "estandar":

        for k, b in enumerate(bits):
            # Producto interno de dos funciones
            Ep = np.sum(Rx[k*p:(k+1)*p] * sinus)
            if Ep > Es/2:
                bitsRx[k] = 1
            else:
                bitsRx[k] = 0

Donde basicamente compara si la potencia de la senal recibida es almenos la mitad de la potencia estandar, de ser este el caso coloca un 1, de ser el caso contrario coloca un 0.

Para ambas señales calculamos su error y su tasa de error (BER)

        err = np.sum(np.abs(bits - bitsRx))
        print('error ', err)

        BER = err/N
        print('BER ',BER)

Por lo tanto para cada SNR se obtienen los siguientes errores y BERs:

* SNR = -2

* SNR = -1

* SNR = 0

* SNR = 1

* SNR = 2

* SNR = 3


## Pregunta 6. Graficar BER versus SNR

Finalmente se graficó BER contra SNR obteniendo asi el siguiente gráfico:

![alt text]()
