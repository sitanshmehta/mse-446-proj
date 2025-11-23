# sliding_window.py
import pandas as pd
import numpy as np


def sliding_window_from_file(path, window_size=20, target_windows=100):
    df = pd.read_csv(path)

    n = len(df)
    if n <= window_size:
        return None

    stride = max(1, (n - window_size) // target_windows)

    windows = []

    for start in range(0, n - window_size + 1, stride):
        window = df.iloc[start:start + window_size]

        features = {
            "IR_mean": window["IR_raw"].mean(),
            "IR_std": window["IR_raw"].std(),
            "IR_var": window["IR_raw"].var(),
            "US_TOF_mean": window["US_pw_us"].mean(),
            "IR_dist_mean": window["IR_distance_cm"].mean(),
            "start_ts": window["timestamp_ms"].iloc[0],
            "end_ts": window["timestamp_ms"].iloc[-1],
        }

        windows.append(features)
        if len(windows) >= target_windows:
            break

    return pd.DataFrame(windows)
