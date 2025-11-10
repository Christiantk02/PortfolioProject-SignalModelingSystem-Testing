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
    white = white_noise(N, p_r)
    noise = np.cumsum(white * (1/fs))

    return noise
        
# Narrowband noise
def narrowband_noise(N, p_mains, fs):
    A = np.sqrt(2 * p_mains)
    t = np.arange(N) / fs
    phi = np.random.uniform(0, 2*np.pi)

    return A * np.cos(2 * np.pi * 50 * t + phi)

# Add noise to ECG
def add_noise(signal, fs, types=("white", "baseline", "narrowband"), factors=(0.01, 0.01, 0.01)):
    N = len(signal)
    noise = np.zeros(N)
    p = 0

    p_signal = np.mean(signal**2)

    if len(types) != len(factors):
        raise ValueError("Length of types and powers must be the same.")

    for t in types:
        if t == "white":
            noise += white_noise(N, factors[p] * p_signal  )
        elif t == "baseline":
            noise += baseline_drift_noise(N, factors[p] * p_signal, fs)
        elif t == "narrowband":
            noise += narrowband_noise(N, factors[p] * p_signal, fs)
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
ecg_noisy = add_noise(ecg, fs, types=("white", "baseline", "narrowband"), factors=(1, 0.5, 0.5))

# Get crossing intervals for both signals
threshold = 0.6

original_intervals = np.array(get_crossing_intervals(ecg, fs, threshold=threshold))*1000
noisy_intervals = np.array(get_crossing_intervals(ecg_noisy, fs, threshold=threshold))*1000

original_BPM = 60 / np.mean(original_intervals) * 1000
noisy_BPM = 60 / np.mean(noisy_intervals) * 1000

# Print the results
print("Original ECG Signal:")
print(f"Number of intervals: {len(original_intervals)}")
print(f"Average BPM: {original_BPM}")

print("\nNoisy ECG Signal:")
print(f"Number of intervals: {len(noisy_intervals)}")
print(f"Average BPM: {noisy_BPM}")

# Plot the signals
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg, label='Original ECG')
plt.axhline(y=threshold, color='red')
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(t, ecg_noisy, label='Noisy ECG', color='orange')
plt.axhline(y=threshold, color='red')
plt.title('Noisy ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
