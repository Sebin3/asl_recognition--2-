import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import cv2
import mediapipe as mp

class Landmarks(BaseModel):
    landmarks: List[float]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # solo para desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de archivos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

# Asegurarse de que el directorio de datos existe
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. Entrena y guarda model.pkl en data/")

model = joblib.load(MODEL_PATH)
print("Modelo cargado:", model)

@app.post("/predict")
async def predict(payload: Landmarks):
    arr = np.array(payload.landmarks, dtype=float).reshape(1, -1)
    # Si tu pipeline necesita preprocesamiento, asegúrate de que model incluya ese pipeline
    pred = model.predict(arr)[0]
    return {"prediction": str(pred)}
