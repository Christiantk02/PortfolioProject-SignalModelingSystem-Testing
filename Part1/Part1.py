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
def add_noise(signal, fs, snr_db, types=("white", "baseline", "narrowband")):
    N = len(signal)
    noise = np.zeros(N)

    p_signal = np.mean(signal**2)
    p_noise = p_signal / (10**(snr_db / 10))
    p_each = p_noise / len(types)

    for t in types:
        if t == "white":
            noise += white_noise(N, p_each)
        elif t == "baseline":
            noise += baseline_drift_noise(N, p_each, fs)
        elif t == "narrowband":
            noise += narrowband_noise(N, p_each, fs)
        else:
           raise ValueError(f"Unknown noise type: {t}")

    p_noise_actual = np.mean(noise**2)
    noise *= np.sqrt(p_noise / p_noise_actual)

    return signal + noise

def evaluate_detection(true_times, detected_times, tolerance_ms=10):
    tolerance_s = tolerance_ms / 1000
    true_hits = 0
    false_hits = 0
    
    for d in detected_times:
        if np.any(np.abs(true_times - d) < tolerance_s):
            true_hits += 1
        else:
            false_hits += 1

    true_ratio = true_hits / len(true_times) if len(true_times) > 0 else 0
    false_ratio = false_hits / len(detected_times) if len(detected_times) > 0 else 0
    
    return true_ratio, false_ratio

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
data = sio.loadmat(r"Data/ecg_data.mat")

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
