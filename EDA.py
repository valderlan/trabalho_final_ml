import json
import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_logger(log_folder: str = "logs", log_file: str = "eda_report.log") -> str:
    """Configura o logger."""
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def remove_outliers_iqr(
    df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, int]:
    """Remove outliers usando IQR."""
    if not numeric_cols:
        return df, 0

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    initial_rows = df.shape[0]
    df_clean = df[condition].copy()
    removed_count = initial_rows - df_clean.shape[0]

    return df_clean, removed_count


def generate_optimized_plots(df: pd.DataFrame, label_col: str, numeric_cols: List[str]):
    """Gera gráficos otimizados para leitura."""

    if label_col in df.columns:
        plt.figure(figsize=(12, 6))
        order = df[label_col].value_counts().index
        ax = sns.countplot(
            x=df[label_col],
            hue=df[label_col],
            order=order,
            palette="viridis",
            legend=False,
        )

        plt.title(f"Distribuição Final da Classe: {label_col}", fontsize=14)
        plt.xlabel("Classe", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)

        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig("figures/class_distribution.png", dpi=300)
        plt.close()

    valid_corr_cols = [c for c in numeric_cols if df[c].std() > 0]
    num_vars = len(valid_corr_cols)

    if num_vars > 1:
        fig_size_w = max(10, num_vars * 0.6)
        fig_size_h = max(8, num_vars * 0.5)

        plt.figure(figsize=(fig_size_w, fig_size_h))

        corr_matrix = df[valid_corr_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9},
        )

        plt.title("Matriz de Correlação (Processada)", fontsize=16, pad=20)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        plt.tight_layout()
        plt.savefig("figures/correlation_matrix.png", dpi=300)
        plt.close()


def exploratory_analysis(
    csv_path: str,
    label_column: str = "label",
    remove_outliers: bool = False,
    drop_rows_with_nan: bool = False,
    drop_cols_with_nan: bool = True,
):

    os.makedirs("logs", exist_ok=True)
    os.makedirs("json", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    setup_logger()
    logging.info("=== Iniciando Análise Otimizada ===")

    report_data = {
        "dataset_info": {},
        "cleaning_summary": {},
        "class_analysis": {},
        "final_status": {},
    }

    df = pd.read_csv(csv_path)

    initial_shape = df.shape
    initial_cols = set(df.columns)

    class_counts_before = {}
    if label_column in df.columns:
        class_counts_before = df[label_column].value_counts().to_dict()

    logging.info(f"Shape Inicial: {initial_shape}")

    if drop_cols_with_nan:
        df = df.dropna(axis=1, how="any")

    if drop_rows_with_nan:
        df = df.dropna(axis=0, how="any")

    df = df.loc[:, (df != 0).any(axis=0)]
    removed_cols = list(initial_cols - set(df.columns))

    outliers_removed = 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if remove_outliers and numeric_cols:
        df, outliers_removed = remove_outliers_iqr(df, numeric_cols)
        logging.info(f"Outliers removidos: {outliers_removed}")
    else:
        logging.info(
            "Remoção de outliers desativada (recomendado para datasets de ataques)."
        )

    class_counts_after = {}
    if label_column in df.columns:
        class_counts_after = df[label_column].value_counts().to_dict()

    class_comparison = []
    all_classes = set(class_counts_before.keys()) | set(class_counts_after.keys())

    logging.info("\n=== Comparativo de Classes ===")
    logging.info(f"{'Classe':<25} | {'Antes':<10} | {'Depois':<10} | {'Perda':<10}")

    for cls in all_classes:
        qtd_before = class_counts_before.get(cls, 0)
        qtd_after = class_counts_after.get(cls, 0)
        diff = qtd_before - qtd_after
        class_comparison.append(
            {
                "class": str(cls),
                "before": int(qtd_before),
                "after": int(qtd_after),
                "loss": int(diff),
            }
        )
        logging.info(
            f"{str(cls):<25} | {qtd_before:<10} | {qtd_after:<10} | {diff:<10}"
        )

    logging.info("\nGerando gráficos otimizados...")
    generate_optimized_plots(df, label_column, numeric_cols)

    final_shape = df.shape

    report_data["dataset_info"] = {
        "initial_rows": int(initial_shape[0]),
        "initial_cols": int(initial_shape[1]),
    }
    report_data["cleaning_summary"] = {
        "removed_columns": removed_cols,
        "outliers_removed": int(outliers_removed),
    }
    report_data["class_analysis"] = class_comparison
    report_data["statistics"] = df.describe().to_dict()

    report_data["final_status"] = {
        "final_rows": int(final_shape[0]),
        "final_columns": int(final_shape[1]),
        "status": "Success",
    }

    logging.info("-" * 40)
    logging.info(f"PROCESSAMENTO CONCLUÍDO")
    logging.info(
        f"SHAPE FINAL DO DATASET: Linhas={final_shape[0]}, Colunas={final_shape[1]}"
    )
    logging.info("-" * 40)

    with open("json/eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    df.to_csv("dataset/dataset_processed_final.csv", index=False)
    logging.info("Dataset salvo em: dataset/dataset_processed_final.csv")

    return df


if __name__ == "__main__":

    exploratory_analysis(
        "dataset/dataset_balanceado.csv",
        label_column="label",
        remove_outliers=False,
        drop_rows_with_nan=False,
        drop_cols_with_nan=True,
    )
