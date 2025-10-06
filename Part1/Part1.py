# Imports
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Functions

# Function to plot the ECG signals
def plotEcg(signal, fs, title="ECG Signal"):

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

def plotNoise(noise, title="Noise"):

    # Plot the signal
    plt.close("all")
    plt.figure(figsize=(10, 4))
    plt.plot(noise, color='blue')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Noise functions

# White noise
def whiteNoise(N, P_white):
    noise = np.random.normal(0, np.sqrt(P_white), N) # Generate white Gaussian noise rng.normal(Middle point, Variance, Length)
    return noise
    
# Baseline drift noise
def baselineDrift(N, P_r, fs):
    noise = np.zeros(N)
    white = whiteNoise(N, P_r)
    for i in range(0, len(white)):
        for j in range(0, i):
            noise[i] += (white[j] * (1/fs))
    return noise
        
        
# Load the ECG file
data = sio.loadmat(r"Part1\Data\ecg_data.mat")

# Extract the ECG signal and sampling frequency and flatten the arrays
ecg = data['s'].flatten()
fs = int(data['fs'].flatten()[0])

