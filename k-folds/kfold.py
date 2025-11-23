# kfold_train.py
import json
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sliding_window import sliding_window_from_file


def load_file_index():
    with open("train_files.json", "r") as f:
        train_files = json.load(f)

    with open("test_files.json", "r") as f:
        test_files = json.load(f)

    # combine for K-fold
    return train_files + test_files


def make_file_df(file_list):
    records = []
    for item in file_list:
        records.append({
            "file": item["file"],
            "geometry": item["geometry"],
            "material": item["material"],
        })
    return pd.DataFrame(records)


def make_windows(file_list):
    X_list = []
    y_list = []

    for item in file_list:
        df = sliding_window_from_file(item["file"])
        if df is None:
            continue

        X_list.append(df)
        y_list.extend([item["material"]] * len(df))   # OR item["geometry"]

    return pd.concat(X_list), y_list


if __name__ == "__main__":
    files = load_file_index()
    file_df = make_file_df(files)

    groups = file_df["file"]  # group by file
    labels = file_df["material"]  # or "geometry"

    gkf = GroupKFold(n_splits=5)

    fold_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(file_df, labels, groups)):
        print(f"\n--- FOLD {fold_idx} ---")

        train_files = file_df.iloc[train_idx].to_dict("records")
        test_files = file_df.iloc[test_idx].to_dict("records")

        X_train, y_train = make_windows(train_files)
        X_test, y_test = make_windows(test_files)

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        fold_accuracies.append(acc)

        print(f"Fold accuracy: {acc:.4f}")

    print("\n===== FINAL RESULTS =====")
    print("Fold Accuracies:", fold_accuracies)
    print("Mean Accuracy:", sum(fold_accuracies) / len(fold_accuracies))
