# split_dataset.py
import os
import json
import random

DATA_BASE = "/Users/sitanshmehta/mse-446-proj/data"
TEST_RATIO = 0.2


def collect_files():
    file_list = []

    for root, dirs, files in os.walk(DATA_BASE):
        if "raw" in root:
            for f in files:
                if f.endswith(".csv"):
                    full_path = os.path.join(root, f)

                    parts = full_path.split(os.sep)
                    geometry = parts[-6]   # e.g., curved/flat
                    material = parts[-5]   # e.g., alu, plastic

                    file_list.append({
                        "file": full_path,
                        "geometry": geometry,
                        "material": material
                    })
    return file_list


def split_files(file_list):
    random.shuffle(file_list)

    n_test = int(len(file_list) * TEST_RATIO)
    test_files = file_list[:n_test]
    train_files = file_list[n_test:]

    return train_files, test_files


if __name__ == "__main__":
    files = collect_files()
    train_files, test_files = split_files(files)

    with open("train_files.json", "w") as f:
        json.dump(train_files, f, indent=4)

    with open("test_files.json", "w") as f:
        json.dump(test_files, f, indent=4)

    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
