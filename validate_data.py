import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RAW_BASE = "data"
WINDOW_SIZE = 20
REPORT_DIR = "validation_reports"


# =============================
# Utility
# =============================
def load_all_processed():
    """
    Walk through data/ and yield (path, df, metadata)
    metadata = {geometry, material, position, filename}
    """
    for root, dirs, files in os.walk(RAW_BASE):
        if "processed" in root:
            for f in files:
                if f.endswith(".csv"):
                    path = os.path.join(root, f)
                    parts = path.split(os.sep)
                    # data / geometry / material / position / processed / file.csv
                    geometry = parts[-5]
                    material = parts[-4]
                    position = parts[-3]

                    df = pd.read_csv(path)
                    yield path, df, {
                        "geometry": geometry,
                        "material": material,
                        "position": position,
                        "filename": f
                    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =============================
# 1. Missing Data Check
# =============================
def check_missing_data(df):
    missing = df.isna().sum()
    return missing


# =============================
# 2. Sampling Stability Check
# =============================
def check_sampling_stability(df):
    if "start_timestamp" not in df.columns:
        return None, None
    ts = df["start_timestamp"].values
    if len(ts) < 2:
        return None, None

    diffs = np.diff(ts)
    return np.mean(diffs), np.std(diffs)


# =============================
# 3. Noise Levels (SNR)
# =============================
def compute_noise(df):
    noise = {}
    for col in ["IR_mean", "IR_std", "IR_variance", "US_TOF_us_mean"]:
        if col in df.columns:
            noise[col] = np.std(df[col])
    return noise


# =============================
# 4. Repeatability Check
# =============================
def compute_repeatability(df_list):
    """
    df_list is all windows from SAME class (geometry, material, position).
    Looks at variation across sessions.
    """
    if len(df_list) < 2:
        return None

    means = []
    for df in df_list:
        if "IR_mean" in df and "US_TOF_us_mean" in df:
            means.append([
                df["IR_mean"].mean(),
                df["US_TOF_us_mean"].mean()
            ])

    means = np.array(means)
    if len(means) < 2:
        return None

    # Pairwise differences
    diffs = []
    for i in range(len(means)):
        for j in range(i+1, len(means)):
            diffs.append(np.linalg.norm(means[i] - means[j]))

    return np.mean(diffs)


# =============================
# 5. Feature Distributions (Per-File Stats)
# =============================
def compute_feature_stats(df):
    stats = {}
    for col in ["IR_mean", "IR_std", "IR_variance", "US_TOF_us_mean"]:
        if col in df.columns:
            stats[col] = (df[col].mean(), df[col].std())
    return stats


# =============================
# 6. Confounders
# =============================
def check_confounds(df):
    results = {}
    if "IR_distance_mean_cm" in df and "US_TOF_us_mean" in df:
        corr = np.corrcoef(df["IR_distance_mean_cm"], df["US_TOF_us_mean"])[0, 1]
        results["distance_vs_ultrasonic_corr"] = corr
    return results


# =============================
# 7. PCA Separability (Within Entire Dataset)
# =============================
def compute_pca_separability(all_features, labels):
    scaler = StandardScaler()
    feats = scaler.fit_transform(all_features)

    pca = PCA(n_components=3)
    embedded = pca.fit_transform(feats)

    # silhouette score (unsupervised separability)
    try:
        score = silhouette_score(embedded, labels)
    except Exception:
        score = None

    return score, pca.explained_variance_ratio_


# =============================
# 8. Outlier Detection
# =============================
def detect_outliers(df):
    if len(df) < 10:
        return None
    cols = ["IR_mean", "IR_std", "IR_variance", "US_TOF_us_mean"]
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        return None

    model = IsolationForest(contamination=0.05)
    X = df[available_cols].values
    preds = model.fit_predict(X)
    outlier_fraction = np.mean(preds == -1)
    return outlier_fraction


# =============================
# 9. Generate Report Per File
# =============================
def write_file_report(path, meta, missing, dt_mean, dt_std, noise, feat_stats, confounds, outliers):
    path_to_raw = Path(path).parent.parent / "raw"
    ensure_dir(path_to_raw)

    path_to_report = path_to_raw / (meta["filename"].replace(".csv", "_report.txt"))


    with open(path_to_report, "w") as f:
        f.write("=== DATA QUALITY REPORT ===\n")
        f.write(f"File: {path}\n")
        f.write(f"Geometry: {meta['geometry']}, Material: {meta['material']}, Position: {meta['position']}\n\n")

        f.write("--- Missing Data ---\n")
        f.write(str(missing) + "\n\n")

        f.write("--- Sampling Stability ---\n")
        f.write(f"Mean Δt = {dt_mean}, Std Δt = {dt_std}\n\n")

        f.write("--- Noise Levels (std of window stats) ---\n")
        f.write(str(noise) + "\n\n")

        f.write("--- Feature Stats ---\n")
        f.write(str(feat_stats) + "\n\n")

        f.write("--- Confounders ---\n")
        f.write(str(confounds) + "\n\n")

        f.write("--- Outliers ---\n")
        f.write(f"Outlier fraction: {outliers}\n\n")

    print(f"[OK] Report saved: {path_to_report}")


# =============================
# Main Pipeline
# =============================
def main():
    all_features = []
    all_labels = []
    class_groups = {}

    # First pass: per-file checks
    for path, df, meta in load_all_processed():

        # 1. missing
        missing = check_missing_data(df)

        # 2. sampling stability
        dt_mean, dt_std = check_sampling_stability(df)

        # 3. noise
        noise = compute_noise(df)

        # 5. feature stats
        feat_stats = compute_feature_stats(df)

        # 6. confounds
        confounds = check_confounds(df)

        # 8. outliers
        outliers = detect_outliers(df)

        # save for PCA
        features_cols = ["IR_mean", "IR_std", "IR_variance", "US_TOF_us_mean"]
        cols_avail = [c for c in features_cols if c in df.columns]
        if cols_avail:
            all_features.append(df[cols_avail].mean().values)
            all_labels.append(f"{meta['material']}_{meta['geometry']}")

        # group for repeatability
        key = (meta["material"], meta["geometry"], meta["position"])
        class_groups.setdefault(key, []).append(df)

        # write individual report
        write_file_report(path, meta, missing, dt_mean, dt_std, noise, feat_stats, confounds, outliers)

    # 4. repeatability
    repeat_report = os.path.join(REPORT_DIR, "repeatability.txt")
    with open(repeat_report, "w") as f:
        f.write("=== REPEATABILITY REPORT ===\n\n")

        for key, dfs in class_groups.items():
            rep = compute_repeatability(dfs)
            f.write(f"{key}: mean inter-session difference = {rep}\n")

    print(f"[OK] Repeatability report saved: {repeat_report}")

    # 7. PCA separability
    if len(all_features) > 3:
        all_features = np.array(all_features)
        silhouette, var_ratio = compute_pca_separability(all_features, all_labels)

        summary_report = os.path.join(REPORT_DIR, "summary_pca.txt")
        with open(summary_report, "w") as f:
            f.write("=== PCA SUMMARY ===\n")
            f.write(f"Silhouette score: {silhouette}\n")
            f.write(f"Variance ratio: {var_ratio}\n")

        print(f"[OK] PCA summary saved: {summary_report}")


if __name__ == "__main__":
    main()
