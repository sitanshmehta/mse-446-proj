from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = "data"
data_rows = []

for root, dirs, files in os.walk(BASE_DIR):
    if "processed" in root:
        for f in files:
            if f.endswith("_processed.csv"):
                filepath = os.path.join(root, f)
                parts = filepath.split(os.sep)

                geometry = parts[1]    # flat / curved
                material = parts[2]    # pla / plastic / fabric / alu
                position = parts[3]    # left / middle / right

                df = pd.read_csv(filepath)
                df["geometry"] = geometry
                df["material"] = material
                df["position"] = position
                data_rows.append(df)

data = pd.concat(data_rows, ignore_index=True)

print("Loaded dataset shape:", data.shape)
print("Unique materials:", data["material"].unique())
print("Unique geometries:", data["geometry"].unique())

FEATURES = ["IR_mean", "IR_variance", "IR_std", "US_TOF_us_mean", "IR_distance_mean_cm"]

X = data[FEATURES]
y_material = data["material"]
y_geometry = data["geometry"]

# Train/test splits
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y_material, test_size=0.2, random_state=42, stratify=y_material
)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X, y_geometry, test_size=0.2, random_state=42, stratify=y_geometry
)

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\n=== Running Grid Search for Material Classification ===")
grid.fit(X_train_m, y_train_m)

best_rf_material = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# Predictions for material
pred_material = best_rf_material.predict(X_test_m)
cm_material = confusion_matrix(y_test_m, pred_material)
acc_material = accuracy_score(y_test_m, pred_material)

print("\nMaterial Classification Accuracy:", acc_material)
print("Confusion Matrix:\n", cm_material)

rf_geom = RandomForestClassifier(n_estimators=300, random_state=42)
rf_geom.fit(X_train_g, y_train_g)

pred_geom = rf_geom.predict(X_test_g)
cm_geom = confusion_matrix(y_test_g, pred_geom)
acc_geom = accuracy_score(y_test_g, pred_geom)

print("\n=== GEOMETRY CLASSIFICATION RESULTS ===")
print("Accuracy:", acc_geom)
print("Confusion Matrix:\n", cm_geom)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(
    cm_material, annot=True, fmt="d", cmap="Blues",
    xticklabels=best_rf_material.classes_,
    yticklabels=best_rf_material.classes_
)
plt.title(f"Material Classification\nAccuracy = {acc_material:.2f}")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.subplot(1, 2, 2)
sns.heatmap(
    cm_geom, annot=True, fmt="d", cmap="Oranges",
    xticklabels=rf_geom.classes_,
    yticklabels=rf_geom.classes_
)
plt.title(f"Geometry Classification\nAccuracy = {acc_geom:.2f}")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()
plt.show()
