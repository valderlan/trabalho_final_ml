import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def setup_logger(log_folder="logs", log_file="preprocessing.log"):
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_folder, log_file), mode="w"),
            logging.StreamHandler(),
        ],
    )


def preprocess_data(input_csv, target_col="label", test_size=0.2):
    setup_logger()
    logging.info("=== Iniciando Pré-processamento para Multi-Modelos ===")

    dirs = ["models", "dataset/processed", "json", "figures"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(input_csv)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    np.save("dataset/processed/X_train.npy", X_train_scaled)
    np.save("dataset/processed/X_test.npy", X_test_scaled)
    np.save("dataset/processed/y_train.npy", y_train)
    np.save("dataset/processed/y_test.npy", y_test)

    metadata = {
        "features": list(X.columns),
        "classes": class_mapping,
        "n_features": X.shape[1],
        "n_classes": len(class_mapping),
    }
    with open("json/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(
        f"Pré-processamento concluído. Features: {metadata['n_features']}, Classes: {metadata['n_classes']}"
    )


if __name__ == "__main__":
    preprocess_data("dataset/dataset_processed_final.csv")
