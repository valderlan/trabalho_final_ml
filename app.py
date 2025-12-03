import json
import os

import joblib
import numpy as np
import onnxruntime as rt
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="ML Multi-Model API", description="API servindo múltiplos modelos via ONNX"
)

loaded_models = {}
scaler = None
metadata = None
eda_report = None
training_metrics = None


@app.on_event("startup")
def load_artifacts():
    global loaded_models, scaler, metadata, eda_report, training_metrics

    if os.path.exists("json/metadata.json"):
        with open("json/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if os.path.exists("json/eda_report.json"):
        with open("json/eda_report.json", "r", encoding="utf-8") as f:
            eda_report = json.load(f)

    if os.path.exists("json/training_metrics.json"):
        with open("json/training_metrics.json", "r", encoding="utf-8") as f:
            training_metrics = json.load(f)

    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
    else:
        print("AVISO: scaler.pkl não encontrado!")

    if os.path.exists("models"):
        for filename in os.listdir("models"):
            if filename.endswith(".onnx"):
                model_name = filename.replace(".onnx", "")
                path = os.path.join("models", filename)
                try:
                    loaded_models[model_name] = rt.InferenceSession(path)
                    print(f"Modelo carregado: {model_name}")
                except Exception as e:
                    print(f"Erro ao carregar {filename}: {e}")

    if "best_model" in loaded_models:
        print("Modelo padrão 'best_model' está pronto.")


class InputData(BaseModel):
    features: list[float]
    model_name: str = "best_model"


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


@app.get("/", response_class=HTMLResponse)
def home():
    """
    Dashboard HTML com link para /docs
    """
    table_rows = ""
    best_model_real_name = "N/A"

    if training_metrics:
        best_model_real_name = training_metrics.get("best_model", "Desconhecido")

        sorted_models = sorted(
            training_metrics.get("models_metrics", []),
            key=lambda x: x["F1_Macro"],
            reverse=True,
        )

        for m in sorted_models:
            is_best = m["Model"] == best_model_real_name
            row_style = "background-color: #d4edda;" if is_best else ""

            table_rows += f"""
            <tr style="{row_style}">
                <td><b>{m['Model']}</b></td>
                <td>{m['Accuracy']}</td>
                <td>{m['F1_Macro']}</td>
                <td>{m['Recall']}</td>
                <td>{m['Precision']}</td>
                <td>{m['Log_Loss']}</td>
                <td>{m['Overfitting_Gap']}</td>
            </tr>
            """

    eda_pretty = (
        json.dumps(eda_report, indent=4) if eda_report else "Relatório não encontrado."
    )

    available_models_html = " ".join(
        [f"<span class='tag'>{k}</span>" for k in loaded_models.keys()]
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            h1, h2 {{ color: #2c3e50; }}
            .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }}
            
            /* Tabela */
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #ddd; text-align: center; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            
            /* Badges e Tags */
            .badge {{ background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
            .tag {{ background-color: #e0e0e0; color: #555; padding: 4px 8px; border-radius: 4px; margin-right: 5px; font-family: monospace; font-size: 0.9em; }}
            
            /* Botão DOCS */
            .btn-docs {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                transition: background 0.3s;
                margin-top: 15px;
            }}
            .btn-docs:hover {{ background-color: #2980b9; }}

            /* EDA */
            pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; max-height: 500px; }}
            details {{ margin-top: 10px; cursor: pointer; }}
            summary {{ font-weight: bold; color: #2980b9; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1> Painel de Controle ML</h1>
            <p><b>Status da API:</b> <span class="badge">ONLINE</span></p>
            <p><b>Modelos Carregados:</b><br>{available_models_html}</p>
            <p><b>Melhor Modelo Geral:</b> <span style="color: #27ae60; font-weight: bold;">{best_model_real_name}</span> (Default)</p>
            
            <a href="/docs" class="btn-docs" target="_blank">📚 Acessar Documentação Swagger (/docs)</a>
        </div>

        <div class="card">
            <h2> Comparativo de Performance (Test Set)</h2>
            <p><small>O modelo destacado em verde é o atual "best_model".</small></p>
            <table>
                <thead>
                    <tr>
                        <th>Modelo (ID)</th>
                        <th>Acurácia</th>
                        <th>F1-Macro</th>
                        <th>Recall</th>
                        <th>Precision</th>
                        <th>Log Loss</th>
                        <th>Gap Overfit</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2> Relatório EDA (Dados Brutos)</h2>
            <details>
                <summary>Clique para expandir o JSON de Análise</summary>
                <pre>{eda_pretty}</pre>
            </details>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
def predict(data: InputData):
    requested_model = data.model_name

    if requested_model not in loaded_models:
        available = list(loaded_models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{requested_model}' não encontrado. Opções disponíveis: {available}",
        )

    session = loaded_models[requested_model]

    if metadata and len(data.features) != metadata["n_features"]:
        raise HTTPException(
            status_code=400, detail=f"Esperado {metadata['n_features']} features."
        )

    try:

        input_arr = np.array(data.features).reshape(1, -1).astype(np.float32)
        input_scaled = scaler.transform(input_arr)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_scaled})

        output_raw = outputs[0]

        if len(outputs) >= 2 and isinstance(outputs[1], list):
            predicted_id = int(output_raw[0])
        else:
            if output_raw.ndim > 1:
                probs = softmax(output_raw)
                predicted_id = int(np.argmax(probs))
            else:
                predicted_id = int(output_raw[0])

        class_name = (
            metadata["classes"].get(str(predicted_id), "Unknown")
            if metadata
            else str(predicted_id)
        )

        return {
            "prediction_id": predicted_id,
            "class_name": class_name,
            "model_used": requested_model,
            "model_type": "ONNX Inference",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
