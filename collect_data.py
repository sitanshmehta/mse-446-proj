import serial
import time
import os
import csv

SERIAL_PORT = "/dev/cu.usbmodem112201"
BAUD_RATE = 115200
DURATION_SEC = 30

geometry = input("Geometry (flat / curved): ").strip().lower()
material = input("Material (PLA / Plastic / Aluminium / Cloth etc.): ").strip().lower()
position = input("Position (left / middle / right): ").strip().lower()

base_dir = "data"
raw_dir = os.path.join(base_dir, geometry, material, position, "raw")
os.makedirs(raw_dir, exist_ok=True)

existing_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
file_index = len(existing_files) + 1
filename = f"data_{file_index}.csv"
filepath = os.path.join(raw_dir, filename)

print(f"\nSaving data to: {filepath}")
print("Collecting data for 30 seconds...\n")

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2) 

with open(filepath, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp_ms", "IR_raw", "IR_distance_cm", "US_pw_us", "US_analog_raw"])

    start_time = time.time()

    while (time.time() - start_time) < DURATION_SEC:
        try:
            line = ser.readline().decode("utf-8").strip()
            if line == "":
                continue

            # t_ms,IR_raw,IR_distance_cm,US_pw_us,US_analog_raw
            parts = line.split(",")

            if len(parts) == 5:
                writer.writerow(parts)
                print(parts)
        except Exception as e:
            print("Error reading line:", e)

print("\nData collection complete!")
print(f"Saved file: {filepath}")
