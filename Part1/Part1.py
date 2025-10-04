# Imports
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


# Functions

# Function to plot the ECG signals
def plotEcg(signal, fs, title):

    # Create a time array from fs
    t = np.arange(len(signal)) / fs

    # Plot the signal
    plt.close("all")
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='red')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Noise functions

# White noise
def whiteNoise(N, P_white):
    rng = np.random.default_rng() # Random number generator
    noise = rng.normal(0, np.sqrt(P_white), N) # Generate white Gaussian noise rng.normal(Middle point, Variance, Length)
    return noise
    
# Baseline drift noise
#def baselineDrift(N, P_r, fs):


# Load the ECG file
data = sio.loadmat(r"Part1\Data\ecg_data.mat")

# print(data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 's', 'fs'])

# Extract the ECG signal and sampling frequency and flatten the arrays
ecg = data['s'].flatten()
fs = int(data['fs'].flatten()[0])

# Plot the ECG signal
plotEcg(ecg, fs, 'ECG Signal')
