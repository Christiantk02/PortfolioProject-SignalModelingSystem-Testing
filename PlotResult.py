import serial
import pandas as pd
import time
import matplotlib.pyplot as plt

port = "COM4"
baud = 115200
dur = 10  # Sek

ser = serial.Serial(port, baud, timeout=0.1)
print("Starting log \n")

raw_list = []
filtered_list = []
bpm_list = []
timestamps = []

start = time.time()

while time.time() - start < dur:
    line = ser.readline().decode(errors='ignore').strip()

    if not line:
        continue

    try:
        raw, filtered, bpm = map(float, line.split(","))

        t = time.time() - start

        raw_list.append(raw)
        filtered_list.append(filtered)
        bpm_list.append(bpm)
        timestamps.append(t)

        print(f"Raw: {raw}, Filtered: {filtered}, BPM: {bpm}")

    except ValueError:
        print(f"Invalid data: {line}")


# === SAVE TO CSV ===
df = pd.DataFrame({
    "t": timestamps,
    "raw": raw_list,
    "filtered": filtered_list,
    "bpm": bpm_list
})

df.to_csv("ecg_recording.csv", index=False)


## === PLOT RESULTS ===
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(timestamps, raw_list, label='Original ECG')
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(timestamps, filtered_list, label='Filtered ECG', color='orange')
plt.title('Filtered ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(timestamps, bpm_list, label='BPM', color='blue')
plt.title('BPM Signal')
plt.xlabel('Time (s)')
plt.ylabel('BPM')
plt.tight_layout()
plt.show()