#######################################################################################################################
# Tarea Especial 6: programa que genere ruidos de colores, efectue la FFT y grafique el IFS.
# Consultar:    https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python
# Brenda Villaseñor 17/12/2025

import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(s,color):
    # Plot the power spectral density of the signal s using numpy functions
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)),color=color)[0]


def noise_psd(N, psd = lambda f: 1):
    # Generate white noise
    X_white = np.fft.rfft(np.random.randn(N))
    # Compute desired power spectral density
    S = psd(np.fft.rfftfreq(N))
    # Normalize S; preserve energy of white noise
    S = S / np.sqrt(np.mean(S**2))
    # Shape white noise to have desired power spectral density
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)

def PSDGenerator(f):
    '''
    Takes a function and returns a function that generates noise depending on the given power spectral density.
    f : function
    N : Length of generated signal
    '''
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    # White noise has a flat power spectral density
    return 1

@PSDGenerator
def blue_noise(f):
    # Blue noise has a power spectral density proportional to f**-2
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    # Violet noise has a power spectral density proportional to f
    return f

@PSDGenerator
def brownian_noise(f):
    # Brownian noise has a power spectral density proportional to 1/f^2
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    # Pink noise has a power spectral density proportional to 1/f
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


# graph the colored noises
plt.figure(figsize=(10, 8))
# List of noises to plot
noises = [white_noise, pink_noise, brownian_noise, blue_noise, violet_noise]
# Corresponding names and colors for plotting
names = ['White Noise', 'Pink Noise', 'Brownian Noise', 'Blue Noise', 'Violet Noise']
colors = ['black', 'magenta', 'brown', 'blue', 'purple']
# Plot each noise and its spectrum
for i, noise in enumerate(noises):
    s = noise(2**14)
    plt.subplot(len(noises), 2, 2*i + 1)
    plt.title(names[i])
    plt.plot(s, color=colors[i])
    plt.subplot(len(noises), 2, 2*i + 2)
    plot_spectrum(s,color=colors[i])
    plt.title(f'Spectrum of {names[i]}')
plt.tight_layout()
plt.show()

# Graph of all spectra together
plt.figure(figsize=(8, 6))
for i, noise in enumerate(noises):
    # Generate noise and plot its spectrum
    s = noise(2**14)
    plot_spectrum(s,color=colors[i])
plt.title('Spectra of Colored Noises')
plt.legend(names)
plt.ylim(10**(-3),10**4)
plt.tight_layout()
plt.show()
#######################################################################################################################