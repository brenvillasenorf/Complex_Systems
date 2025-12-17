
# Trabajo sobre Bifurcaciones II

## Mapeo de Henón
# Usando el mapeo de Henón $$x_{n+1}=1-ax_n^2+y_n$$ $$y_{n+1}=bx_n$$

# 1) Graficamos el atractor reconstruido en $x_{n+1}$ vs $x_n$ para $a=1.4$ y $b=3$.


import numpy as np
import matplotlib.pyplot as plt

# creamos la función para el mapeo de Henón
def henon(x,y,a,b):
    return 1-a*x**2+y, b*x

# graficamos el mapa de henón
a=1.4
b=0.3
x=np.zeros(10000)
y=np.zeros(10000)
x[0]=0.1
y[0]=0.1
for i in range(9999):
    x[i+1], y[i+1]=henon(x[i],y[i],a,b)

plt.plot(x,y,',', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa de Henon')
plt.show()

# graficamos x_n+1 vs x_n
a=1.4
b=0.3
x_values = np.zeros(10000)
y_values = np.zeros(10000)
x_values[0] = 0.1
y_values[0] = 0.1

for i in range(9999):
    x_values[i+1], y_values[i+1] = henon(x_values[i], y_values[i], a, b)

# graficamos x_n+1 vs x_n
plt.figure(figsize=(10, 6))
plt.plot(x_values[:-1], x_values[1:], ',')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+1}$')
plt.title('$x_{n+1}$ vs $x_n$ for Henon Map')
plt.show()

"""2) Graficamos el diagrama de bifurcaciones para $b=3$ y $a$ en [1,1.5]."""

# Variamos a y fijamos b
a_values = np.linspace(1.0, 1.5, 10000)
b = 0.3

transient_iterations = 800
collection_iterations = 100

a_bifurcation = []
x_bifurcation = []

# Iniciamos valores
x = 0.1
y = 0.1

for a in a_values:
    # Descartamos puntos del transient
    for _ in range(transient_iterations):
        x, y = henon(x, y, a, b)

    # Colleccionamos los puntos para la bifurcacion
    overflow = False
    for _ in range(collection_iterations):
        x, y = henon(x, y, a, b)
        if abs(x) > 1000 or abs(y) > 1000:
            overflow = True
            break
        a_bifurcation.append(a)
        x_bifurcation.append(x)

    if overflow:
        continue

# Graficamos
plt.figure(figsize=(10, 6))
plt.plot(a_bifurcation, x_bifurcation, ',', alpha=0.5)
plt.xlabel('a')
plt.ylabel('x')
plt.title('Bifurcation Diagram for Henon Map (b=0.3)')
plt.ylim(-1.5, 1.5)
plt.xlim(1, 1.5)
plt.axvline(x=1.026, color='r', linestyle='-')
plt.axvline(x=1.051, color='r', linestyle='-')
plt.axvline(x=1.059, color='r', linestyle='-')
plt.grid(True)
plt.show()

# Imprimimos el exponente de Feigenbaum
delta_1 = (1.051 - 1.026) / (1.059 - 1.051)
print(f'Exponente de Feigenbaum aproximado: {delta_1}')

# Si bien, observamos que para $r<1.1$ el mapeo exhibe una ruta al caos 
# de doblamiento de periodo. No obstante, para parámetros mayores se observa la ruta de intermitencia.

# ## Ecuación de Lorenz
# $$\frac{dx}{dt}=\sigma(y-x)$$
# $$\frac{dy}{dt}=x(\rho-z)-y$$
# $$\frac{dz}{dt}=xy-\beta z$$

# 1) Graficamos el atractor en 3D usando $x(t)$, $y(t)$ y $z(t)$.

# creamos la función de la ecuacion de lorenz
def lorenz(x, y, z, sigma, rho, beta, delta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    x = x + delta*dx
    y = y + delta*dy
    z = z + delta*dz
    return x, y, z

# inicializamos parámetros
sigma =10
rho =30
beta = 2.6666
N = 5000
delta = 0.011

# guardamos valores para graficar el atractor de Lorenz
x_lorenz = []
y_lorenz = []
z_lorenz = []

x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

x[0] = 1
y[0] = 1
z[0] = 1

x_lorenz.append(x[0])
y_lorenz.append(y[0])
z_lorenz.append(z[0])

for i in range(N-1):
    x[i+1], y[i+1], z[i+1] = lorenz(x[i], y[i], z[i], sigma, rho, beta, delta)
    x_lorenz.append(x[i+1])
    y_lorenz.append(y[i+1])
    z_lorenz.append(z[i+1])

# graficamos en 3D el atractor de Lorenz
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,15))
fig.patch.set_facecolor('white') # Changed to white
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_lorenz, y_lorenz, z_lorenz, color='crimson', alpha=0.8, linewidth=0.5)

ax.set_xlabel('x',color='black') # Changed to black
ax.set_ylabel('y',color='black') # Changed to black
ax.set_zlabel('z',color='black') # Changed to black
plt.title('Lorenz Attractor',color='black') # Changed to black

# Mejoramos el dise;o del grafico
ax.grid(animated=True)
ax.set_facecolor('white') # Changed to white
ax.tick_params(axis='both', colors='black') # Changed to black

ax.xaxis.label.set_color('black') # Changed to black
ax.yaxis.label.set_color('black') # Changed to black
ax.zaxis.label.set_color('black') # Changed to black

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.show()

# Obtenemos los valores para graficar el atractor con x(t),x(t+1),x(t+2)
x_lorenz_aprox = []
y_lorenz_aprox = []
z_lorenz_aprox = []

x, y, z = 1, 1, 1

for i in range(N):
    x, y, z = lorenz(x, y, z, sigma, rho, beta, delta)
    x_lorenz_aprox.append(x)
    y_lorenz_aprox.append(y)
    z_lorenz_aprox.append(z)

# Creamos las secuencias para  x(t), x(t+1), y x(t+2)
if len(x_lorenz_aprox) > 2:
    x_lorenz_1 = x_lorenz_aprox[:-2]
    x_lorenz_2 = x_lorenz_aprox[1:-1]
    x_lorenz_3 = x_lorenz_aprox[2:]

# graficamos en 3D x(t),x(t+1),x(t+2)
fig = plt.figure(figsize=(10,20))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_lorenz_1, x_lorenz_2, x_lorenz_3, color='crimson', alpha=0.8, linewidth=0.5)
ax.set_xlabel('X(t)',color='black')
ax.set_ylabel('X(t+1)',color='black')
ax.set_zlabel('x(t+2)',color='black', labelpad=10)

# Improve aesthetics
ax.grid(animated=True)
ax.set_facecolor('white')
ax.tick_params(axis='both', colors='black')

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.zaxis.label.set_color('black')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.title('Atractor de Lorenz Reconstruido',color='black',fontsize=20)
plt.show()

# Graficamo el diagrama de bifurcaciones variando rho en [0,50] y fijando sigma=10 y beta=8/3
sigma = 10
beta = 8/3
rho_values = np.linspace(0, 200, 10000)
rho_bifurcation = []
x_bifurcation = []

# Iniciamos valores
x = 1.0
y = 1.0
z = 1.0

transient_iterations = 800
collection_iterations = 100

for rho in rho_values:
    # Descartamos puntos del transient
    for _ in range(transient_iterations):
        x, y, z = lorenz(x, y, z, sigma, rho, beta, delta)
    # Colleccionamos los puntos para la bifurcacion
    overflow = False
    for _ in range(collection_iterations):
        x, y, z = lorenz(x, y, z, sigma, rho, beta, delta)
        if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
            overflow = True
            break
        rho_bifurcation.append(rho)
        x_bifurcation.append(x)
    if overflow:
        continue
# Graficamos
plt.figure(figsize=(10, 6))
plt.plot(rho_bifurcation, x_bifurcation, ',', alpha=0.5)
plt.xlabel('rho')
plt.ylabel('x')
plt.title('Bifurcation Diagram for Lorenz System (sigma=10, beta=8/3)')
plt.ylim(-20, 20)
plt.xlim(0, 50)
plt.grid(True)
plt.show()
