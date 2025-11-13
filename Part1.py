# === IMPORTS ===o
import numpy as np

# === FUNCTIONS ===

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
def add_noise(signal, fs, snr_db):
    N = len(signal)
    noise = np.zeros(N)

    p_signal = np.mean(signal**2)
    p_noise = p_signal / (10**(snr_db / 10))

    p_white = p_noise * 0.7
    p_baseline = p_noise * 0.1
    p_mains = p_noise * 0.2

    noise += white_noise(N, p_white)
    noise += baseline_drift_noise(N, p_baseline, fs)
    noise += narrowband_noise(N, p_mains, fs)

    p_noise_actual = np.mean(noise**2)
    noise *= np.sqrt(p_noise / p_noise_actual)

    return signal + noise

# Evaluate crossings detected
def evaluate_detection(true_times, detected_times, tolerance_ms=5):
    tolerance_s = tolerance_ms / 1000.0
    
    true_hits = 0
    used = np.zeros(len(detected_times), dtype=bool)

    for t in true_times:
        diffs = np.abs(detected_times - t)
        idx = np.argmin(diffs)

        if diffs[idx] <= tolerance_s and not used[idx]:
            true_hits += 1
            used[idx] = True

    false_hits = np.sum(~used)

    true_ratio = true_hits / len(true_times) if len(true_times) > 0 else 0
    false_ratio = false_hits / len(detected_times) if len(detected_times) > 0 else 0

    return true_ratio, false_ratio

# Get crossing intervals
def get_crossings(signal, fs, threshold=0.0):
    N = len(signal)
    crossings = []
    above_threshold = True

    for n in range(N):
        if signal[n] > threshold and not above_threshold:
            above_threshold = True
            crossing = n/fs
            crossings.append(crossing)
        elif signal[n] <= threshold and above_threshold:
            above_threshold = False

    return crossings

