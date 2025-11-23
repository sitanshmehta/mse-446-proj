import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "data"
data_rows = []

for root, dirs, files in os.walk(BASE_DIR):
    if "processed" in root:
        for f in files:
            if f.endswith("_processed.csv"):
                filepath = os.path.join(root, f)
                parts = filepath.split(os.sep)

                geometry = parts[1]
                material = parts[2]
                position = parts[3]

                df = pd.read_csv(filepath)
                df["geometry"] = geometry
                df["material"] = material
                df["position"] = position

                data_rows.append(df)

data = pd.concat(data_rows, ignore_index=True)

print("Loaded dataset shape:", data.shape)
print(data.head())
print("Unique materials:", data["material"].unique())

FEATURES = [
    "IR_mean",
    "IR_variance",
    "IR_std",
    "US_TOF_us_mean",
    "IR_distance_mean_cm"
]

y_geom = data["geometry"]
y_mat = data["material"]
X = data[FEATURES]

X_train, X_test, y_geom_train, y_geom_test = train_test_split(
    X, y_geom, test_size=0.2, random_state=42, stratify=y_geom
)

_, _, y_mat_train, y_mat_test = train_test_split(
    X, y_mat, test_size=0.2, random_state=42, stratify=y_mat
)

rf_geom = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_geom.fit(X_train, y_geom_train)
pred_geom = rf_geom.predict(X_test)

print("\n=== GEOMETRY CLASSIFICATION (Flat vs Curved) ===")
print("Accuracy:", accuracy_score(y_geom_test, pred_geom))
cm_geom = confusion_matrix(y_geom_test, pred_geom)
print("Confusion Matrix:\n", cm_geom)

rf_mat = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_mat.fit(X_train, y_mat_train)
pred_mat = rf_mat.predict(X_test)

print("\n=== MATERIAL CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_mat_test, pred_mat))
cm_mat = confusion_matrix(y_mat_test, pred_mat)
print("Confusion Matrix:\n", cm_mat)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_geom, annot=True, fmt="d", cmap="Blues",
            xticklabels=rf_geom.classes_, yticklabels=rf_geom.classes_,
            ax=axes[0])
axes[0].set_title("Geometry Classification (Flat vs Curved)")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=rf_mat.classes_, yticklabels=rf_mat.classes_,
            ax=axes[1])
axes[1].set_title("Material Classification")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.show()