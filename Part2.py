# === IMPORTS ===
import numpy as np

# === FUNCTIONS ===

# Lowpass filter
def lowpass_fir(cutoff, fs, N=101):
    fc = cutoff / fs
    n = np.arange(N)

    w = np.hamming(N)
    h = 2 * fc * np.sinc((fc * 2) * (n - (N - 1) / 2))
    
    h = h * w
    h = h / np.sum(h)

    return h

# Highpass filter
def highpass_fir(cutoff, fs, N=101):
    delta = np.zeros(N)
    delta[N//2] = 1
    return delta - lowpass_fir(cutoff, fs, N)

# Bandstop filter
def bandstop_fir(f_low, f_high, fs, N=101):
    return lowpass_fir(f_low, fs, N) + highpass_fir(f_high, fs, N)

# Full filter pipeline
def filter_pipeline(x, fs, hp_cut=0.5, bs_low=49, bs_high=51, lp_cut=35, N=101):
    h_hp = highpass_fir(hp_cut, fs, N)
    x_hp = np.convolve(x, h_hp, mode='same')

    h_bs = bandstop_fir(bs_low, bs_high, fs, N)
    x_bs = np.convolve(x_hp, h_bs, mode='same')

    h_lp = lowpass_fir(lp_cut, fs, N)
    x_clean = np.convolve(x_bs, h_lp, mode='same')

    return x_clean