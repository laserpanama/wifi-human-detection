import serial
import time
import os

PORT = "/dev/ttyUSB0"
BAUD = 921600

folder = "../data/experiments"
os.makedirs(folder, exist_ok=True)

filename = f"{folder}/csi_{int(time.time())}.csv"

ser = serial.Serial(PORT, BAUD)

with open(filename, "w") as f:
    while True:
        line = ser.readline().decode(errors="ignore")
        f.write(line)
        f.flush()
        print(line.strip())
