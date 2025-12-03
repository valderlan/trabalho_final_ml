import json
import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset


def setup_logger(log_folder="logs", log_file="training.log"):
    os.makedirs(log_folder, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_folder, log_file), mode="w", encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc_input_dim = 16 * (input_dim // 2)
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PyTorchCNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=10, num_classes=2, epochs=10, lr=0.001):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.model = SimpleCNN(self.input_dim, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


def calculate_metrics(y_true, y_pred, y_prob, model_name, dataset_type="Test"):
    """Calcula todas as métricas solicitadas."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        ll = log_loss(y_true, y_prob) if y_prob is not None else -1.0
    except:
        ll = -1.0

    return {
        "Model": model_name,
        "Dataset": dataset_type,
        "Accuracy": round(acc, 4),
        "F1_Macro": round(f1, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "MCC": round(mcc, 4),
        "Log_Loss": round(ll, 4) if ll != -1.0 else "N/A",
    }


def plot_confusion_matrix(y_true, y_pred, model_name, classes):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.savefig(f"figures/cm_{model_name}.png")
    plt.close()


def plot_feature_importance(model, model_name, feature_names):
    """
    Gera gráfico de importância das features se o modelo suportar.
    Salva em figures/feature_importance_{model_name}.png
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        df_imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_imp.head(15), x="Importance", y="Feature", palette="viridis"
        )
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(f"figures/feature_importance_{model_name}.png")
        plt.close()
        logging.info(f"Gráfico de Feature Importance salvo para {model_name}")
    else:
        pass


def export_model_onnx(model_obj, model_name, n_features, model_type="sklearn"):
    """Exporta o modelo para ONNX e salva na pasta models/"""
    onnx_path = f"models/{model_name}.onnx"

    try:
        if model_type == "sklearn":
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onx = convert_sklearn(model_obj, initial_types=initial_type)
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())

        elif model_type == "torch":
            dummy_input = torch.randn(1, n_features, dtype=torch.float32).to(
                model_obj.device
            )
            model_obj.model.eval()
            torch.onnx.export(
                model_obj.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["float_input"],
                output_names=["output"],
                dynamic_axes={
                    "float_input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
        logging.info(f"Modelo salvo em ONNX: {onnx_path}")
        return True
    except Exception as e:
        logging.error(f"Erro ao exportar {model_name}: {e}")
        return False


def main():
    setup_logger()
    logging.info("=== Iniciando Treinamento com Grid Search ===")

    for d in ["models", "json", "figures"]:
        os.makedirs(d, exist_ok=True)

    try:
        X_train = np.load("dataset/processed/X_train.npy")
        X_test = np.load("dataset/processed/X_test.npy")
        y_train = np.load("dataset/processed/y_train.npy")
        y_test = np.load("dataset/processed/y_test.npy")

        with open("json/metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        return

    n_features = metadata["n_features"]
    n_classes = metadata["n_classes"]
    classes_list = [metadata["classes"][str(i)] for i in range(n_classes)]
    feature_names = metadata.get("features", [f"feat_{i}" for i in range(n_features)])

    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1, 10]},
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "criterion": ["gini", "entropy"],
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(random_state=42, n_jobs=1),
            "params": {
                "n_estimators": [100, 200],
                "criterion": ["gini", "entropy"],
                "min_samples_split": [2, 5],
            },
        },
        "MLP_Sklearn": {
            "model": MLPClassifier(max_iter=500, random_state=42),
            "params": {
                "hidden_layer_sizes": [(64, 32), (100,)],
                "alpha": [0.0001, 0.001],
            },
        },
        "CNN_PyTorch": {
            "model": PyTorchCNNWrapper(
                input_dim=n_features, num_classes=n_classes, epochs=15
            ),
            "params": {},
        },
    }

    full_report = []
    best_overall_f1 = -1
    best_overall_model_name = ""

    for name, config in models_config.items():
        logging.info(f"\n>> Processando: {name}")

        model = config["model"]
        params = config["params"]

        if params and "PyTorch" not in name:
            logging.info(f"Iniciando Grid Search para {name}...")
            grid = GridSearchCV(model, params, cv=5, scoring="f1_macro", n_jobs=1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logging.info(f"Melhores parâmetros {name}: {grid.best_params_}")
        else:
            logging.info(f"Treinando {name} (sem Grid Search)...")
            model.fit(X_train, y_train)
            best_model = model

        y_pred_train = best_model.predict(X_train)
        y_prob_train = (
            best_model.predict_proba(X_train)
            if hasattr(best_model, "predict_proba")
            else None
        )
        metrics_train = calculate_metrics(
            y_train, y_pred_train, y_prob_train, name, "Train"
        )

        y_pred_test = best_model.predict(X_test)
        y_prob_test = (
            best_model.predict_proba(X_test)
            if hasattr(best_model, "predict_proba")
            else None
        )
        metrics_test = calculate_metrics(y_test, y_pred_test, y_prob_test, name, "Test")

        gap = metrics_train["Accuracy"] - metrics_test["Accuracy"]
        metrics_test["Overfitting_Gap"] = round(gap, 4)

        full_report.append(metrics_test)

        logging.info(f"Métricas Detalhadas para {name}: {json.dumps(metrics_test)}")

        plot_confusion_matrix(y_test, y_pred_test, name, classes_list)
        plot_feature_importance(best_model, name, feature_names)

        model_type = "torch" if "PyTorch" in name else "sklearn"
        export_model_onnx(best_model, name, n_features, model_type)

        if metrics_test["F1_Macro"] > best_overall_f1:
            best_overall_f1 = metrics_test["F1_Macro"]
            best_overall_model_name = name

    report_path_json = "json/training_metrics.json"
    final_json = {
        "best_model": best_overall_model_name,
        "best_f1_macro": best_overall_f1,
        "models_metrics": full_report,
    }
    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)
    logging.info(f"Relatório JSON salvo em: {report_path_json}")

    df_results = pd.DataFrame(full_report)

    cols_order = [
        "Model",
        "Accuracy",
        "F1_Macro",
        "Recall",
        "Precision",
        "MCC",
        "Log_Loss",
        "Overfitting_Gap",
    ]
    df_md = df_results[cols_order].sort_values(by="F1_Macro", ascending=False)

    report_path_md = "relatorio_comparativo.md"
    with open(report_path_md, "w", encoding="utf-8") as f:
        f.write("# Relatório Comparativo de Modelos de ML\n\n")
        f.write(
            f"**Melhor Modelo Geral:** {best_overall_model_name} (F1: {best_overall_f1})\n\n"
        )
        try:
            f.write(df_md.to_markdown(index=False))
        except ImportError:
            f.write(df_md.to_string(index=False))

    logging.info(f"Tabela Markdown salva em: {report_path_md}")

    df_results.to_csv("dataset/processed/model_comparison_results.csv", index=False)

    best_model_path = f"models/{best_overall_model_name}.onnx"
    target_path = "models/best_model.onnx"
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, target_path)
        logging.info(f"Melhor modelo copiado para: {target_path}")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_results,
        x="Model",
        y="F1_Macro",
        hue="Model",
        legend=False,
        palette="viridis",
    )
    plt.title("Comparativo de Modelos (F1-Macro - Test Set)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/final_comparison_f1.png")
    plt.close()


if __name__ == "__main__":
    main()
