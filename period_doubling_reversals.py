###############################################################################################################################################
# Programar y reproducir los resultados de L. Stone.
# (Period doubling reversals. Nature 1993) FIG. 1 y reportar.
#
# Brenda Villaseñor Feixas 17/12/2025

import numpy as np
import matplotlib.pyplot as plt

# FIG 1 of L. Stone's paper says:
# "The bifurcation diagram for the logistic maps:
# a) x_n+1 = x_n*exp(r*(K-x_n))
# b) x_n+1 = x_n*exp(r*(1-x_n))+h 
# h : number of immigrants per generation (h = 0.06)
# K : carrying capacity
# r : growth rate
# r varies from 1.8 to 4.0 in steps of 0.001
# iterations = 700
# last 200 points are plotted
# initial condition x0 = 0.5

def logistic_map_A(r, K, x):
    return x * np.exp(r*(K - x))

def logistic_map_B(r, x, h):
    return x * np.exp(r * (1 - x)) + h

# Parameters
r_values = np.arange(1.8, 4.0, 0.001)
iterations = 700
last_points = 200
K = 1.0
h = 0.06
x0 = 0.5 
x_A = np.zeros((len(r_values), iterations))
x_B = np.zeros((len(r_values), iterations))
x_A[:, 0] = x0
x_B[:, 0] = x0

# Iterate the maps
for i in range(1, iterations):
    x_A[:, i] = logistic_map_A(r_values, K, x_A[:, i-1])
    x_B[:, i] = logistic_map_B(r_values, x_B[:, i-1], h)


    
# Plots
plt.figure(figsize=(12, 6))
# Plot for map A
plt.subplot(1, 2, 1)
plt.title("Bifurcation Diagram for Map A")
plt.xlabel("Growth Rate (r)")
plt.ylabel("Population (x)")
for i, r in enumerate(r_values):
    plt.plot([r]*last_points, x_A[i, -last_points:], ',k', alpha=0.25)
# Plot for map B
plt.subplot(1, 2, 2)    
plt.title("Bifurcation Diagram for Map B")
plt.xlabel("Growth Rate (r)")
plt.ylabel("Population (x)")
for i, r in enumerate(r_values):
    plt.plot([r]*last_points, x_B[i, -last_points:], ',k', alpha=0.25)
plt.tight_layout()
plt.show()


# Report:
# Ambos diagramas de bifurcación generados para los mapas logísticos A y B exhiben la característica ruta al caos de doblamiento de período descrita por L. Stone.
# En el mapa A, a medida que aumenta la tasa de crecimiento r, observamos que la población se estabiliza en puntos fijos, seguidos de bifurcaciones que conducen a ciclos periódicos y eventualmente a un comportamiento caótico.
# El mapa B muestra dinámicas similares, pero la presencia del término de inmigración h introduce una complejidad adicional en la estructura de bifurcación.
# En ambos casos, los diagramas reflejan la sensibilidad a las condiciones iniciales y la transición hacia el caos, confirmando los resultados reportados en el artículo original.
# Si se coloca como condición inicial de x0=1 no se observa comportamiento caótico en el mapa A, pues se escogió como capacidad de carga K=1.0, por lo que la población no refleja cambios y la exponencial se vuelve 1.