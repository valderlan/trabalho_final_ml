import pandas as pd


def undersample_large_csv(input_csv, target_column, output_csv, chunksize=200_000):
    print("Passo 1/2 — Contando classes...")
    class_counts = {}

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        counts = chunk[target_column].value_counts().to_dict()
        for cls, n in counts.items():
            class_counts[cls] = class_counts.get(cls, 0) + n

    print("Distribuição original:")
    print(class_counts)

    minority_count = min(class_counts.values())
    print(f"\nClasse minoritária tem {minority_count} registros.")

    collected = {cls: 0 for cls in class_counts.keys()}

    open(output_csv, "w").close()
    header_written = False

    print("\nPasso 2/2 — Realizando undersampling em streaming...")

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        parts = []

        for cls in class_counts.keys():
            cls_mask = chunk[target_column] == cls
            cls_chunk = chunk[cls_mask]

            available = len(cls_chunk)
            needed = minority_count - collected[cls]

            if needed <= 0:
                continue

            if available > needed:
                sampled = cls_chunk.sample(needed, random_state=42)
            else:
                sampled = cls_chunk

            collected[cls] += len(sampled)
            parts.append(sampled)

        if parts:
            balanced_chunk = pd.concat(parts)

            balanced_chunk.to_csv(
                output_csv, mode="a", header=not header_written, index=False
            )
            header_written = True

    print("\n✔ Balanceamento concluído!")
    print(f"Arquivo salvo em: {output_csv}")


if __name__ == "__main__":
    balanced_df = undersample_large_csv(
        input_csv="dataset/ddos.csv",
        target_column="label",
        output_csv="dataset/dataset_balanceado.csv",
    )
