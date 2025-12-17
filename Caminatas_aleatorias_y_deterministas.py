"""Caminantes_aleatorios_y_deterministas.ipynb
Original file by Brenda Villaseñor is located at
    https://colab.research.google.com/drive/14IXVc8Hqbj_U3sDSJWIOknU4B2C_Akzw

#   Caminanta aleatoria
##  *Movimiento Browniano*
    Es un proceso estocástico que describe el movimiento aleatorio de una partícula.
"""

# importamos librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import random
from scipy.stats import gaussian_kde

### Ejemplo en 1D  ###

def caminata_aleatoria_1D(n_pasos, i_pos, n_trayectorias=1000):
  '''
  n_pasos = número de pasos
  i_pos = posición inicial
  n_trayectorias = número de caminatas independientes para estimar la distribución
  '''

  # Seleccionamos la posición inicial y guardamos la posición en una lista
  x_i = i_pos
  x_positions = [x_i]

  # Creamos una lista para guardar los valores del desplazamiento medio cuadrático
  MSD = []
  msd = 0

  # Definimos las posiciones correspondientes a cada dirección
  for i in range(n_pasos):
    # Escogemos un número aleatorio para indicar la dirección de movimiento
    # r = 1 (derecha)
    # r = -1 (izquierda)
    r = np.random.choice([-1, 1])  # Escalar (no array) para evitar ambigüedad
    if r == 1:
      x_i += 1
    else:
      x_i -= 1
    # Guardamos la nueva posición
    x_positions.append(x_i)
    # Calculamos y guardamos el desplazamiento cuadrático medio
    msd += (x_positions[i] - i_pos) ** 2
    MSD.append(msd * 1 / (i + 1))

  # Graficamos la caminata aleatoria y su distribución de probabilidad
  plt.figure(figsize=(15,5))
  plt.subplot(1,3,1)
  plt.title("Caminata aleatoria en 1D")
  plt.xlabel("Tiempo")
  plt.ylabel("Posición")
  plt.plot(x_positions)
  plt.tight_layout()
  plt.subplot(1,3,2)
  plt.title("Desplazamiento cuadrático medio")
  plt.xlabel("Tiempo")
  plt.ylabel("MSD")
  plt.loglog(np.arange(1, len(MSD) + 1), MSD)
  # Graficamos la identidad y=x en log log
  plt.loglog(np.arange(1, len(MSD) + 1), np.arange(1, len(MSD) + 1), linestyle='--', color='red', label='y=x')
  plt.tight_layout()

  # Graficamos la distribución de probabilidad 
  pasos_ensemble = np.random.choice([-1, 1], size=(n_trayectorias, n_pasos))
  posiciones_finales = i_pos + pasos_ensemble.sum(axis=1)
  mu, sigma = norm.fit(posiciones_finales)
  
  plt.subplot(1,3,3)
  plt.hist(posiciones_finales, bins=50, density=True, alpha=0.6, color='g', label='Posición final (ensemble)')
  x_min, x_max = plt.xlim()
  x_grid = np.linspace(x_min, x_max, 200)
  p = norm.pdf(x_grid, mu, sigma)
  plt.plot(x_grid, p, 'k', linewidth=2, label=f'Ajuste Normal: mu={mu:.2f}, sigma={sigma:.2f}')
  plt.title("Distribución de posiciones finales")
  plt.xlabel("Posición (x)")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)

  return plt.show()

### Ejemplo en 2D  ###

def caminata_aleatoria_2D(n_pasos, i_pos, n_trayectorias=1000):
  '''
  n_pasos = número de pasos
  i_pos = posición inicial
  n_trayectorias = número de caminatas independientes para estimar la distribución
  '''
  # Seleccionamos la posición inicial y guardamos la posición
  # en una lista
  x_i = i_pos[0]
  y_i = i_pos[1]
  x = [x_i]
  y = [y_i]

  # Creamos una lista para guardar los valores del desplazamiento
  # medio cuadrático
  MSD_x = []
  MSD_y = []
  msd_x = 0
  msd_y = 0

  # Definimos las posiciones correspondientes a cada dirección
  for i in range(n_pasos):
    # Escogemos un número aleatorio para indicar la dirección
    # de movimiento
    # r = 1 (arriba)
    # r = 2 (abajo)
    # r = 3 (derecha)
    # r = 4 (izquierda)
      r = random.randint(1,4)

      if r == 1:
          y_i += 1
      elif r == 2:
          y_i -= 1
      elif r == 3:
          x_i += 1
      else:
          x_i -= 1
      x.append(x_i)
      y.append(y_i)

      msd_x += (x[i]-i_pos[0])**2
      msd_y += (y[i]-i_pos[1])**2
      MSD_x.append(msd_x*1/(i+1))
      MSD_y.append(msd_y*1/(i+1))

  # Graficamos la caminata aleatoria
  # La distribución de probabilidad
  # El desplazamiento cuadrático medio
  plt.figure(figsize=(14,10))
  plt.subplot(2,2,1)
  plt.title("Caminata aleatoria en 2D")
  plt.xlabel("Posición x")
  plt.ylabel("Posición y")
  plt.plot(x,y)
  plt.subplot(2,2,2)
  plt.title("Desplazamiento cuadrático medio")
  plt.xlabel("Tiempo")
  plt.ylabel("MSD")
  plt.loglog(MSD_x, label="MSD x")
  plt.loglog(MSD_y, label="MSD y")
  plt.legend()

  # Graficamos la distribución de probabilidad 
  steps = np.random.choice([0, 1, 2, 3], size=(n_trayectorias, n_pasos))
  dx = np.zeros_like(steps, dtype=int)
  dy = np.zeros_like(steps, dtype=int)
  dx[steps == 0] = 1
  dx[steps == 1] = -1
  dy[steps == 2] = 1
  dy[steps == 3] = -1
  posiciones_finales_x = i_pos[0] + dx.sum(axis=1)
  posiciones_finales_y = i_pos[1] + dy.sum(axis=1)
  mu_x, sigma_x = norm.fit(posiciones_finales_x)
  mu_y, sigma_y = norm.fit(posiciones_finales_y)

  
  # Ajuste en x
  plt.subplot(2,2,3)
  plt.hist(posiciones_finales_x, bins=50, density=True, alpha=0.6, color='g', label='Posición final (x)')
  x_min, x_max = plt.xlim()
  x_grid = np.linspace(x_min, x_max, 200)
  p_x = norm.pdf(x_grid, mu_x, sigma_x)
  plt.plot(x_grid, p_x, 'k', linewidth=2, label=f'Ajuste Normal x: mu={mu_x:.2f}, sigma={sigma_x:.2f}')
  plt.title("Distribución de posiciones finales (x)")
  plt.xlabel("Posición")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)
  
  # Ajuste en y
  plt.subplot(2,2,4)
  plt.hist(posiciones_finales_y, bins=50, density=True, alpha=0.4, color='r', label='Posición final (y)')
  y_min, y_max = plt.ylim()
  y_grid = np.linspace(y_min, y_max, 200)
  p_y = norm.pdf(y_grid, mu_y, sigma_y)
  plt.plot(y_grid, p_y, 'blue', linewidth=2, label=f'Ajuste Normal y: mu={mu_y:.2f}, sigma={sigma_y:.2f}')
  plt.title("Distribución de posiciones finales (y)")
  plt.xlabel("Posición")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)
  plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)

  return plt.show()


"""# Caminata determinista

## Caminantes de Lévy
Caminata con un patrón de Ley de Potencias $P(l)=l^{-\alpha}$.

Realizamos para $\alpha=1.05,1.5,2.0,2.5,6.0$.
"""

def caminata_determinista_1d(alpha, n_pasos, i_pos, n_trayectorias=1000):
  '''
  alpha = parámetro de la ley de potencias
  n_pasos = número de pasos
  i_pos = posición inicial
  n_trayectorias = número de caminatas independientes para estimar la distribución
  '''

  # Seleccionamos la posición inicial y guardamos la posición
  # en una lista
  x_0 = i_pos
  x = [x_0]

  # Creamos una lista para guardar los valores del desplazamiento
  # medio cuadrático
  MSD = []
  msd = 0

  # Creamos una lista que guarde la historia

  # Definimos las posiciones correspondientes a cada dirección
  for i in range(n_pasos):
    # Definimos la probabilidad de paso de acuerdo a la
    # distribución de probabilidad P(l)=l**(-a)
      # El tamaño de paso sigue la Ley de potencias
      l = np.random.pareto(alpha)
      # La dirección de paso es aleatoria
      dir = np.random.choice([-1, 1])
      x_i = x[i] + dir * l
      x.append(x_i)
      msd += (x_i-i_pos)**2
      MSD.append(msd*1/(i+1))

  # Graficamos la caminata aleatoria y su
  # distribución de probabilidad
  plt.figure(figsize=(15,5))
  plt.subplot(1,3,1)
  plt.title(f'Caminata determinista en 1D para alpha = {alpha}')
  plt.xlabel("Tiempo")
  plt.ylabel("Posición")
  plt.plot(x)
  plt.tight_layout()
  plt.subplot(1,3,2)
  plt.title("Desplazamiento cuadrático medio (Log-Log)")
  plt.xlabel("Tiempo")
  plt.ylabel("MSD")
  plt.loglog(np.arange(1, len(MSD) + 1), MSD)
  plt.tight_layout()

  # Graficamos la distribución de probabilidad usando posiciones finales de muchas trayectorias independientes
  pasos = np.random.pareto(alpha, size=(n_trayectorias, n_pasos))
  direcciones = np.random.choice([-1, 1], size=(n_trayectorias, n_pasos))
  desplazamientos = pasos * direcciones
  posiciones_finales = i_pos + desplazamientos.sum(axis=1)
  kde = gaussian_kde(posiciones_finales)
  plt.subplot(1,3,3)
  plt.hist(posiciones_finales, bins=50, density=True, alpha=0.6, color='g', label='Posición final')
  x_min, x_max = plt.xlim()
  x_grid = np.linspace(x_min, x_max, 200)
  plt.plot(x_grid, kde(x_grid), 'k', linewidth=2, label='KDE')
  plt.title("Distribución de posiciones finales")
  plt.xlabel("Posición (x)")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()

  return plt.show()


def caminata_determinista_2d(alpha, n_pasos, i_pos, n_trayectorias=1000):
  '''
  alpha = parámetro de la ley de potencias
  n_pasos = número de pasos
  i_pos = posición inicial
  n_trayectorias = número de caminatas independientes para estimar la distribución
  '''

  # Seleccionamos la posición inicial y guardamos la posición
  # en una lista
  x_0 = i_pos[0]
  y_0 = i_pos[1]
  x = [x_0]
  y = [y_0]

  # Creamos una lista para guardar los valores del desplazamiento
  # medio cuadrático
  MSD_x = []
  MSD_y = []
  msd_x = 0
  msd_y = 0

  # Creamos una lista que guarde la historia

  # Definimos las posiciones correspondientes a cada dirección
  for i in range(n_pasos):
    # Definimos la probabilidad de paso de acuerdo a la
    # distribución de probabilidad P(l)=l**(-a)
      # El tamaño de paso sigue la Ley de potencias
      l = np.random.pareto(alpha)
      # La dirección de paso es aleatoria
      dir = np.random.uniform(0,2*np.pi)
      x_i = x[i] + (l * np.cos(dir))
      y_i = y[i] + (l * np.sin(dir))
      x.append(x_i)
      y.append(y_i)
      msd_x += (x_i-i_pos[0])**2
      msd_y += (y_i-i_pos[1])**2
      MSD_x.append(msd_x*1/(i+1))
      MSD_y.append(msd_y*1/(i+1))

  # Graficamos la caminata aleatoria y su
  # distribución de probabilidad

  plt.figure(figsize=(15,7))
  plt.subplot(2,2,1)
  plt.title(f'Caminata determinista en 1D para alpha = {alpha}')
  plt.xlabel("Tiempo")
  plt.ylabel("Posición")
  plt.plot(x,y)
  plt.subplot(2,2,2)
  plt.title("Desplazamiento cuadrático medio (Log-Log)")
  plt.xlabel("Tiempo")
  plt.ylabel("MSD")
  plt.loglog(np.arange(1, len(MSD_x) + 1), MSD_x, label="MSD x")
  plt.loglog(np.arange(1, len(MSD_y) + 1), MSD_y, label="MSD y")
  plt.loglog(np.arange(1, len(MSD_x) + 1), np.arange(1, len(MSD_x) + 1), linestyle='--', color='red', label='y=x')
  plt.legend()

  # Graficamos la distribución de probabilidad 
  pasos = np.random.pareto(alpha, size=(n_trayectorias, n_pasos))
  angulos = np.random.uniform(0, 2 * np.pi, size=(n_trayectorias, n_pasos))
  dx = pasos * np.cos(angulos)
  dy = pasos * np.sin(angulos)
  posiciones_finales_x = i_pos[0] + dx.sum(axis=1)
  posiciones_finales_y = i_pos[1] + dy.sum(axis=1)
  
  # Estimación de densidad con KDE (Kernel Density Estimation) para una curva suave
  kde_x = gaussian_kde(posiciones_finales_x)
  kde_y = gaussian_kde(posiciones_finales_y)
  
  
  plt.subplot(2,2,3)
  plt.hist(posiciones_finales_x, bins=50, density=True, alpha=0.6, color='g', label='Posición final (x)')
  x_min, x_max = plt.xlim()
  x_grid = np.linspace(x_min, x_max, 200)
  plt.plot(x_grid, kde_x(x_grid), 'k', linewidth=2, label='KDE x')
  plt.title("Distribución de posiciones finales (x)")
  plt.xlabel("Posición")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)
  
  plt.subplot(2,2,4)
  plt.hist(posiciones_finales_y, bins=50, density=True, alpha=0.4, color='r', label='Posición final (y)')
  y_min, y_max = plt.ylim()
  y_grid = np.linspace(y_min, y_max, 200)
  plt.plot(y_grid, kde_y(y_grid), 'blue', linewidth=2, label='KDE y')
  plt.title("Distribución de posiciones finales (y)")
  plt.xlabel("Posición")
  plt.ylabel("Densidad de Probabilidad")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()

  return plt.show()


#caminata_aleatoria_1D(10000,0)
#caminata_aleatoria_2D(1000, [0,0])
#caminata_determinista_1d(2.4, 100000, 0)
#caminata_determinista_2d(3, 100000, [0,0])

alfas = [1.05, 1.5, 2.0, 2.5, 6.0]
for alpha in alfas:
  caminata_determinista_1d(alpha, 100000, 0)
  caminata_determinista_2d(alpha, 100000, [0,0])

