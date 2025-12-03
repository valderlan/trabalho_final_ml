import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt
import joblib
import numpy as np
import json
import os

app = FastAPI(title="ML Multi-Model API", description="API servindo o melhor modelo (Sklearn ou PyTorch) via ONNX")

model_session = None
scaler = None
metadata = None

@app.on_event("startup")
def load_artifacts():
    global model_session, scaler, metadata
    try:
        with open("json/metadata.json", "r") as f:
            metadata = json.load(f)
        
        scaler = joblib.load("models/scaler.pkl")
        model_session = rt.InferenceSession("models/best_model.onnx")
        print("Artefatos carregados. Modelo pronto.")
    except Exception as e:
        print(f"ERRO FATAL: {e}")

class InputData(BaseModel):
    features: list[float]

def softmax(x):
    """Calcula softmax para logits (caso venha do PyTorch)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.post("/predict")
def predict(data: InputData):
    if not model_session:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    if len(data.features) != metadata["n_features"]:
        raise HTTPException(status_code=400, detail=f"Esperado {metadata['n_features']} features.")

    try:
        # 1. Preprocess
        input_arr = np.array(data.features).reshape(1, -1).astype(np.float32)
        input_scaled = scaler.transform(input_arr)
        
        # 2. Inference
        input_name = model_session.get_inputs()[0].name
        # ONNX Runtime output handling
        outputs = model_session.run(None, {input_name: input_scaled})
        
        # 3. Post-process
        # Sklearn ONNX geralmente retorna [Labels, Probabilities]
        # PyTorch ONNX geralmente retorna [Logits]
        
        output_raw = outputs[0]
        
        if len(outputs) >= 2 and isinstance(outputs[1], list): # Padrão Sklearn (Label, Probs)
             # Na maioria dos casos convert_sklearn retorna labels diretos no indice 0
             predicted_id = int(output_raw[0])
             # As probabilidades estariam em outputs[1], mas é uma lista de dicts muitas vezes
        else:
            # Padrão PyTorch (Logits) ou Sklearn simplificado
            # Se for logits, aplicamos argmax
            if output_raw.ndim > 1:
                probs = softmax(output_raw)
                predicted_id = int(np.argmax(probs))
            else:
                predicted_id = int(output_raw[0])

        class_name = metadata["classes"].get(str(predicted_id), "Unknown")

        return {
            "prediction_id": predicted_id,
            "class_name": class_name,
            "model_type": "ONNX Inference"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)