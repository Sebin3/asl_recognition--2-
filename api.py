# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

class Landmarks(BaseModel):
    landmarks: List[float]

app = FastAPI(
    title="ASL Recognition API",
    description="API para reconocimiento de lenguaje de señas americano",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeo de números a letras (ajusta según tu modelo)
LETTER_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

MODEL_PATH = "data/model.pkl"
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("Modelo cargado:", model)

@app.get("/")
def read_root():
    return {
        "message": "API de reconocimiento de señas activa",
        "endpoints": {
            "predict": "/predict - POST para predecir una letra",
            "health": "/health - GET para verificar el estado"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model": str(model)}

@app.post("/predict")
async def predict(payload: Landmarks):
    try:
        arr = np.array(payload.landmarks, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        # Si tu modelo devuelve números, convertirlos a letras
        letter = LETTER_MAP.get(pred, str(pred))
        return {
            "prediction": letter,
            "confidence": float(max(model.predict_proba(arr)[0]))
        }
    except Exception as e:
        return {"error": str(e)}
