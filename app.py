import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import onnxruntime as rt
import joblib
import numpy as np
import json
import os

app = FastAPI(title="ML Multi-Model API", description="API servindo múltiplos modelos via ONNX")


loaded_models = {} 
scaler = None
metadata = None
eda_report = None
training_metrics = None

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
    
    loaded_models.clear()
    print("Aplicação encerrada e memória limpa.")

app = FastAPI(
    title="ML Multi-Model API", 
    description="API servindo múltiplos modelos via ONNX",
    lifespan=lifespan
)

class InputData(BaseModel):
    features: list[float]
    model_name: str = "best_model"

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Dashboard HTML Completo
    """

    table_rows_models = ""
    best_model_real_name = "N/A"
    
    if training_metrics:
        best_model_real_name = training_metrics.get("best_model", "Desconhecido")
        sorted_models = sorted(training_metrics.get("models_metrics", []), key=lambda x: x['F1_Macro'], reverse=True)

        for m in sorted_models:
            is_best = m['Model'] == best_model_real_name
            row_style = "background-color: #d4edda;" if is_best else ""

            mcc = m.get('MCC', '-')
            log_loss = m.get('Log_Loss', '-')
            
            table_rows_models += f"""
            <tr style="{row_style}">
                <td><b>{m['Model']}</b></td>
                <td>{m['Accuracy']}</td>
                <td>{m['F1_Macro']}</td>
                <td>{m['Recall']}</td>
                <td>{m['Precision']}</td>
                <td>{mcc}</td>
                <td>{log_loss}</td>
                <td>{m['Overfitting_Gap']}</td>
            </tr>
            """

    stats_rows_html = ""
    dataset_stats_html = "<p>Dados não disponíveis.</p>"
    cleaning_info_html = ""
    classes_rows_html = ""
    
    if eda_report:

        initial_rows = eda_report.get("dataset_info", {}).get("initial_rows", 0)
        final_rows = eda_report.get("final_status", {}).get("final_rows", 0)
        final_cols = eda_report.get("final_status", {}).get("final_columns", 0)
        removed_cols = eda_report.get("cleaning_summary", {}).get("removed_columns", [])
        
        dataset_stats_html = f"""
        <div class="stats-grid">
            <div class="stat-box">
                <h3>Total Amostras</h3>
                <p>{final_rows:,}</p>
                <small>Original: {initial_rows:,}</small>
            </div>
            <div class="stat-box">
                <h3>Features Finais</h3>
                <p>{final_cols}</p>
            </div>
             <div class="stat-box">
                <h3>Colunas Removidas</h3>
                <p>{len(removed_cols)}</p>
            </div>
        </div>
        """
        
        if removed_cols:
            cleaning_info_html = f"<p><b>Colunas ignoradas:</b> <span style='color: #e74c3c; font-family: monospace;'>{', '.join(removed_cols)}</span></p>"

        class_analysis = eda_report.get("class_analysis", [])
        for cls in class_analysis:
            classes_rows_html += f"<tr><td style='text-align: left;'>{cls['class']}</td><td>{cls['after']}</td><td>{cls['loss']}</td></tr>"

        statistics = eda_report.get("statistics", {})
        for feat_name, stats in statistics.items():

            mean = f"{stats.get('mean', 0):.4f}"
            std = f"{stats.get('std', 0):.4f}"
            mini = f"{stats.get('min', 0):.4f}"
            median = f"{stats.get('50%', 0):.4f}" 
            maxi = f"{stats.get('max', 0):.4f}"
            
            stats_rows_html += f"""
            <tr>
                <td style="text-align: left; font-weight: bold;">{feat_name}</td>
                <td>{mean}</td>
                <td>{std}</td>
                <td>{mini}</td>
                <td style="background-color: #f0f8ff;">{median}</td>
                <td>{maxi}</td>
            </tr>
            """

    available_models_html = " ".join([f"<span class='tag'>{k}</span>" for k in loaded_models.keys()])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f4f9; color: #333; padding-bottom: 50px; }}
            .container {{ max_width: 1200px; margin: 0 auto; padding: 20px; }}
            
            h1, h2 {{ color: #2c3e50; margin-top: 0; }}
            
            /* Cards */
            .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }}
            
            /* Stats Grid */
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px; }}
            .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }}
            .stat-box h3 {{ margin: 0; font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; }}
            .stat-box p {{ margin: 10px 0 0; font-size: 1.8em; font-weight: bold; color: #2c3e50; }}
            .stat-box small {{ color: #95a5a6; font-size: 0.8em; }}

            /* Tabelas */
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #ddd; text-align: center; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            
            /* Badges */
            .badge {{ background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
            .tag {{ background-color: #e0e0e0; color: #555; padding: 4px 8px; border-radius: 4px; margin-right: 5px; font-family: monospace; font-size: 0.9em; }}
            .btn-docs {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 15px; }}
            
            /* Scroll para tabela grande */
            .table-container {{ overflow-x: auto; max-height: 500px; }}
            
            details {{ margin-top: 10px; cursor: pointer; }}
            summary {{ font-weight: bold; color: #2980b9; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h1> Painel de Controle ML</h1>
                        <p><b>Status:</b> <span class="badge">ONLINE</span> &nbsp; | &nbsp; <b>Modelos Disponíveis:</b> {len(loaded_models)}</p>
                    </div>
                    <div>
                        <a href="/docs" class="btn-docs" target="_blank"> Testar API (/docs)</a>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p><b>Melhor Modelo:</b> <span style="color: #27ae60; font-weight: bold;">{best_model_real_name}</span></p>
                    <small>Carregados: {available_models_html}</small>
                </div>
            </div>

            <div class="card">
                <h2> Performance dos Modelos (Test Set)</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Modelo</th>
                                <th>Acurácia</th>
                                <th>F1-Macro</th>
                                <th>Recall</th>
                                <th>Precision</th>
                                <th>MCC</th>
                                <th>Log Loss</th>
                                <th>Overfit Gap</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows_models}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2> Estatísticas do Dataset (EDA)</h2>
                {dataset_stats_html}
                <div style="margin-top: 20px;">{cleaning_info_html}</div>

                <details>
                    <summary> Ver Estatísticas Descritivas (Tabela Detalhada)</summary>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th style="text-align: left;">Feature</th>
                                    <th>Média</th>
                                    <th>Desvio Padrão</th>
                                    <th>Mínimo</th>
                                    <th>Mediana (50%)</th>
                                    <th>Máximo</th>
                                </tr>
                            </thead>
                            <tbody>
                                {stats_rows_html}
                            </tbody>
                        </table>
                    </div>
                </details>

                <details>
                    <summary> Ver Distribuição de Classes</summary>
                    <table>
                        <thead>
                            <tr>
                                <th style="text-align: left;">Classe</th>
                                <th>Qtd Final</th>
                                <th>Perda</th>
                            </tr>
                        </thead>
                        <tbody>
                            {classes_rows_html}
                        </tbody>
                    </table>
                </details>
            </div>
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
            detail=f"Modelo '{requested_model}' não encontrado. Opções disponíveis: {available}"
        )
    
    session = loaded_models[requested_model]

    if metadata and len(data.features) != metadata["n_features"]:
        raise HTTPException(status_code=400, detail=f"Esperado {metadata['n_features']} features.")

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

        class_name = metadata["classes"].get(str(predicted_id), "Unknown") if metadata else str(predicted_id)

        return {
            "prediction_id": predicted_id,
            "class_name": class_name,
            "model_used": requested_model,
            "model_type": "ONNX Inference"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)