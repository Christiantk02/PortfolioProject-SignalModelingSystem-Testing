# Imports
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Functions

# White noise
def white_noise(N, p_white):
    noise = np.random.normal(0, np.sqrt(p_white), N)

    return noise
    
# Baseline drift noise
def baseline_drift_noise(N, p_r, fs):
    noise = np.zeros(N)
    white = white_noise(N, p_r)
    for i in range(0, len(white)):
        for j in range(0, i):
            noise[i] += (white[j] * (1/fs))

    return noise
        
# Narrowband noise
def narrowband_noise(N, p_mains, fs):
    A = np.sqrt(2 * p_mains)
    t = np.arange(N) / fs
    phi = np.random.uniform(0, 2*np.pi)

    return A * np.cos(2 * np.pi * 50 * t + phi)

# Add noise to ECG
def add_noise(signal, fs, types=("white", "baseline", "narrowband"), powers=(0.01, 0.01, 0.01)):
    N = len(signal)
    noise = np.zeros(N)
    p = 0

    if len(types) != len(powers):
        raise ValueError("Length of types and powers must be the same.")

    for t in types:
        if t == "white":
            noise += white_noise(N, powers[p])
        elif t == "baseline":
            noise += baseline_drift_noise(N, powers[p], fs)
        elif t == "narrowband":
            noise += narrowband_noise(N, powers[p], fs)
        else:
           raise ValueError(f"Unknown noise type: {t}")
        p += 1 

    return signal + noise

# Get crossing intervals
def get_crossing_intervals(signal, fs, threshold=0.0):
    N = len(signal)
    intervals = []
    above_threshold = True
    last_crossing = 0.0

    for n in range(N):
        if signal[n] > threshold and not above_threshold:
            above_threshold = True
            crossing = n/fs
            intervals.append(crossing - last_crossing)
            last_crossing = crossing
        elif signal[n] <= threshold and above_threshold:
            above_threshold = False

    return intervals

# Main Code

# Load the ECG file
data = sio.loadmat(r"Part1\Data\ecg_data.mat")

# Extract the ECG signal and sampling frequency and flatten the arrays
ecg = data['s'].flatten()
fs = int(data['fs'].flatten()[0])

# Add noise to the ECG signal
ecg_noisy = add_noise(ecg, fs, types=("white", "baseline", "narrowband"), powers=(0.03, 0.01, 0.01))

# Get crossing intervals for both signals
treshold = 0.6

orginal_intervals = np.array(get_crossing_intervals(ecg, fs, threshold=treshold))*1000
noisy_intervals = np.array(get_crossing_intervals(ecg_noisy, fs, threshold=treshold))*1000

orginal_BPM = 60 / np.average(orginal_intervals) * 1000
noisy_BPM = 60 / np.average(noisy_intervals) * 1000

# Print the results
print("Original ECG Signal:")
print(f"Number of intervals: {len(orginal_intervals)}")
print(f"Avrage BPM: {orginal_BPM}")

print("\nNoisy ECG Signal:")
print(f"Number of intervals: {len(noisy_intervals)}")
print(f"Avrage BPM: {noisy_BPM}")

# Plot the signals
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg, label='Original ECG')
plt.axhline(y=treshold, color='red')
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(t, ecg_noisy, label='Noisy ECG', color='orange')
plt.axhline(y=treshold, color='red')
plt.title('Noisy ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
