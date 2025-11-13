# === IMPORTS ===
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import Part1 as p1
import Part2 as p2


# === RUN TEST ===

# Load the ECG file
data = sio.loadmat(r"Data/ecg_data.mat")

# Extract the ECG signal and sampling frequency and flatten the arrays
ecg = data['s'].flatten()
fs = int(data['fs'].flatten()[0])

# Add noise to the ECG signal
ecg_noisy = p1.add_noise(ecg, fs, 1)

# Clean noisy signal
ecg_cleaned = p2.filter_pipeline(ecg_noisy, fs)

# Get threshold of max signal
threshold = np.percentile(ecg_cleaned, 98)

# Get crossings for all signals
original_crossings = np.array(p1.get_crossings(ecg, fs, threshold=threshold))
noisy_crossings = np.array(p1.get_crossings(ecg_noisy, fs, threshold=threshold))
cleaned_crossings = np.array(p1.get_crossings(ecg_cleaned, fs, threshold=threshold))


# === RESULTS ===

# Print results for each signal
print("Original ECG Signal:")
print(f"Original Crossings: {np.round(original_crossings, 3)}")
print(f"Number of original crosisngs:  {len(original_crossings)}")

print("\nNoisy ECG Signal:")
print(f"Noisy Crossings: {np.round(noisy_crossings, 3)}")
print(f"Number og noisy crossings: {len(noisy_crossings)}")

true_ratio, false_ratio = p1.evaluate_detection(original_crossings, noisy_crossings)

print(f"True ratio, False ratio noisy signal:{round(true_ratio, 2)}, {round(false_ratio, 2)}")

print("\nCleaned ECG Signal:")
print(f"Cleaned Crossings: {np.round(cleaned_crossings, 3)}")
print(f"Number og noisy crossings: {len(cleaned_crossings)}")

true_ratio, false_ratio = p1.evaluate_detection(original_crossings, cleaned_crossings)

print(f"True ratio, False ratio cleaned signal:{round(true_ratio, 2)}, {round(false_ratio, 2)}")


# === PLOT ===

# Plot each signal
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, ecg, label='Original ECG')
plt.axhline(y=threshold, color='red')
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, ecg_noisy, label='Noisy ECG', color='orange')
plt.axhline(y=threshold, color='red')
plt.title('Noisy ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, ecg_cleaned, label='Cleaned ECG', color='blue')
plt.axhline(y=threshold, color='red')
plt.title('Cleaned ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()