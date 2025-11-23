import os
import pandas as pd
import numpy as np

RAW_BASE = "data"
WINDOW_SIZE = 20


def process_csv(raw_path, processed_path):
    """Process one raw CSV into windowed statistical features."""

    df = pd.read_csv(raw_path)

    required = ["timestamp_ms", "IR_raw", "IR_distance_cm", "US_pw_us", "US_analog_raw"]
    if not all(col in df.columns for col in required):
        print(f"Skipping {raw_path} (missing columns)")
        return

    windows = []
    num_windows = len(df) // WINDOW_SIZE

    for i in range(num_windows):
        window = df.iloc[i*WINDOW_SIZE : (i+1)*WINDOW_SIZE]

        ir_raw_vals = window["IR_raw"].values
        us_pw_vals = window["US_pw_us"].values

        features = {
            "start_timestamp": int(window["timestamp_ms"].iloc[0]),
            "end_timestamp": int(window["timestamp_ms"].iloc[-1]),

            # IR features
            "IR_mean": float(np.mean(ir_raw_vals)),
            "IR_variance": float(np.var(ir_raw_vals)),
            "IR_std": float(np.std(ir_raw_vals)),

            # Ultrasonic (TOF averaged)
            "US_TOF_us_mean": float(np.mean(us_pw_vals)),
            "IR_distance_mean_cm": float(np.mean(window["IR_distance_cm"])),
        }

        windows.append(features)

    processed_df = pd.DataFrame(windows)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed and saved: {processed_path}")


def process_all_data():
    """Walk through the entire folder tree and process all raw files."""
    for root, dirs, files in os.walk(RAW_BASE):
        if "raw" in root:
            for f in files:
                if f.endswith(".csv"):
                    raw_path = os.path.join(root, f)
                    processed_dir = root.replace("raw", "processed")
                    processed_filename = f.replace(".csv", "_processed.csv")
                    processed_path = os.path.join(processed_dir, processed_filename)
                    process_csv(raw_path, processed_path)


if __name__ == "__main__":
    process_all_data()
