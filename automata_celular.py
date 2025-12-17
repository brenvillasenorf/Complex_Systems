# # Programar un automata celular en 1D:
# # - Ancho de la cadena binaria: 100 elementos.
# # - Recibir como entrada el número de regla en decimal y transformar a binario.
# # - Condición a la frontera: periódica.
# # - Evaluar el valor de la entropia de Shannon-Kolgomorov (y su promedio temporal) para cada paso de tiempo y graficar.
# # - Explorar, al menos, las siguientes reglas: 18, 22, 30, 45, 110
# # - Condiciones inicials: semilla, periódica 1/5 y aleatoria equiprobable

# Brenda Villaseñor Feixas

# # Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import random


# Función para convertir el número decimal a una lista binaria de 8 bits
def decimal_a_binario(d):
    '''d: número decimal de la regla'''
    return [int(x) for x in np.binary_repr(d, width=8)][::-1]

# Función para aplicar la regla del autómata celular
def aplicar_regla(c, rb):
    '''c: cadena actual
       rb: regla binaria'''
    # Crear una nueva cadena para el siguiente estado
    nc = np.zeros_like(c)
    l = len(c)
    for i in range(l):
        izquierda = c[(i - 1) % l]
        centro = c[i]
        derecha = c[(i + 1) % l]
        index = izquierda * 4 + centro * 2 + derecha
        nc[i] = rb[index]
    return nc

# Función para calcular la entropía de Shannon-Kolmogorov
def calcular_entropia(c):
    '''c: cadena actual'''
    # Calcular la distribución de probabilidad de los estados
    valores, counts = np.unique(c, return_counts=True)
    prob = counts / len(c)
    return entropy(prob, base=2)

# Función principal para ejecutar el autómata celular
def ejecutar_automata(rd, ci, t):
    '''rd: regla decimal
       ci: lista de condiciones iniciales
       t: número de pasos de tiempo a simular
       rb: regla binaria'''
    # Convertir la regla decimal a binaria
    rb = decimal_a_binario(rd)
    # Diccionario para almacenar los resultados
    res = {}
    # Ejecutar para cada condición inicial
    for cond in ci:
        # Si la condición es semilla, periódica 1/5 o aleatoria
        if cond == 'semilla':
            c = np.zeros(100, dtype=int)
            # Establecer un único 1 en el centro
            c[50] = 1
        elif cond == 'periodica_1_5':
            # Crear una cadena periódica con 1 cada 5 posiciones
            c = np.array([1 if i % 5 == 0 else 0 for i in range(100)], dtype=int)
        elif cond == 'aleatoria':
            # Crear una cadena aleatoria equiprobable
            c = np.random.choice([0, 1], size=100)
        # Almacenar la historia y entropías
        his = [c.copy()]
        ent = [calcular_entropia(c)]
        ent_promedio = []
        ep = 0
        # Ejecutar el autómata celular por los pasos de tiempo
        for _ in range(t):
            c = aplicar_regla(c, rb)
            his.append(c.copy())
            ent.append(calcular_entropia(c))
            ep += ent[-1]
            ent_promedio.append(ep / (len(ent)))
        # Guardar los resultados
        res[cond] = {
            'historia': np.array(his),
            'entropias': np.array(ent),
            'entropia_promedio': np.array(ent_promedio)
        }

    return res

# Parámetros de la simulación
reglas = [18, 22, 30, 45, 110]
ci = ['semilla', 'periodica_1_5', 'aleatoria']
t = 100

# Ejecutar la simulación y graficar los resultados
for regla in reglas:
    resultados = ejecutar_automata(regla, ci, t)

    for cond, datos in resultados.items():
        hist = datos['historia']
        ent = datos['entropias']
        ent_promedio = datos['entropia_promedio']

        # Graficar la evolución del autómata celular
        plt.figure(figsize=(10, 6))
        plt.imshow(hist, cmap='binary', interpolation='nearest')
        plt.title(f'Autómata Celular - Regla {regla} - Condición: {cond}')
        plt.xlabel('Posición')
        plt.ylabel('Paso de Tiempo')
        plt.show()

        # Graficar la entropía a lo largo del tiempo
        plt.figure(figsize=(10, 4))
        plt.plot(ent, label='Entropía de Shannon-Kolmogorov')
        plt.plot(ent_promedio, label='Entropía Promedio', linestyle='--')
        plt.title(f'Entropía - Regla {regla} - Condición: {cond}')
        plt.xlabel('Paso de Tiempo')
        plt.ylabel('Entropía (bits)')
        plt.legend()
        plt.show()


# # Para revisar las reglas del autómata https://www.cs.us.es/~fsancho/Blog/posts/Automatas_Celulares.md.html

