import serial
import matplotlib.pyplot as plt
from time import sleep

port = "COM4"
baud = 115200

ser = serial.Serial(port, baud, timeout=0.1)
print("Starting Serial Monitor \n")

while True:
    line = ser.readline().decode(errors='ignore').strip()

    if not line:
        continue

    try:
        raw, filtered, bpm = map(float, line.split(","))

        print(f"BPM: {bpm}")

        sleep(0.1)

    except ValueError:
        print(f"Invalid data: {line}")