import argparse
import json
import os
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split


def write_mltable(folder: str, filename: str):
    # MLTable that points to one file
    with open(os.path.join(folder, "MLTable"), "w", encoding="utf-8") as f:
        f.write(
f"""paths:
  - file: ./{filename}
transformations:
  - read_delimited:
      delimiter: ","
      encoding: "utf8"
      header: all_files_same_headers
"""
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)

    parser.add_argument("--target_col", type=str, default="house_affiliation")
    parser.add_argument("--delimiter", type=str, default=",")

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stratify", type=int, default=1)  # 1=true, 0=false

    parser.add_argument("--out_train", type=str, required=True)
    parser.add_argument("--out_test", type=str, required=True)

    args = parser.parse_args()

    # locate csv
    csv_path = os.path.join(args.input_folder, "data.csv")
    if not os.path.exists(csv_path):
        csvs = glob(os.path.join(args.input_folder, "*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in input folder: {args.input_folder}")
        csv_path = csvs[0]

    df = pd.read_csv(csv_path, sep=args.delimiter)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Columns: {list(df.columns)}")

    # ✅ Drop columns that should never be features (prevent leakage)
    # - character_name is basically an ID / label proxy
    # - character_id is also an ID
    leakage_cols = ["character_id", "character_name"]
    drop_cols = [c for c in leakage_cols if c in df.columns]

    y = df[args.target_col].astype(str)
    X = df.drop(columns=[args.target_col] + drop_cols, errors="ignore")

    strat = None
    if args.stratify == 1:
        vc = y.value_counts()
        too_small = vc[vc < 2]
        if len(too_small) == 0:
            strat = y
            print("✅ Using stratified split.")
        else:
            strat = None
            print("⚠️ Stratify disabled: at least one class has < 2 samples.")
            print("Classes with too few samples:", too_small.to_dict())

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strat
    )

    # One-hot encode categoricals on TRAIN, then align TEST
    X_train = pd.get_dummies(X_train_raw, dummy_na=True)
    X_test = pd.get_dummies(X_test_raw, dummy_na=True)

    # Align columns so train/test match exactly
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Save outputs
    os.makedirs(args.out_train, exist_ok=True)
    os.makedirs(args.out_test, exist_ok=True)

    X_train_path = os.path.join(args.out_train, "X_train.csv")
    y_train_path = os.path.join(args.out_train, "y_train.csv")
    X_test_path = os.path.join(args.out_test, "X_test.csv")
    y_test_path = os.path.join(args.out_test, "y_test.csv")

    X_train.to_csv(X_train_path, index=False, encoding="utf-8")
    y_train.to_frame(name=args.target_col).to_csv(y_train_path, index=False, encoding="utf-8")

    X_test.to_csv(X_test_path, index=False, encoding="utf-8")
    y_test.to_frame(name=args.target_col).to_csv(y_test_path, index=False, encoding="utf-8")

    # Save metadata for reproducibility & inference compatibility
    with open(os.path.join(args.out_train, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"columns": list(X_train.columns)}, f, ensure_ascii=False, indent=2)

    classes = sorted(y.unique().tolist())
    with open(os.path.join(args.out_train, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    # Add MLTable definitions (point to the main files)
    write_mltable(args.out_train, "X_train.csv")
    write_mltable(args.out_test, "X_test.csv")

    print("✅ Done")
    print("Target:", args.target_col)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Num classes:", len(classes))


if __name__ == "__main__":
    main()
