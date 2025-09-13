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
MODEL_DIR = '/app/data'  # Usar ruta absoluta consistente
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

# Verificar que el modelo existe
if not os.path.isfile(MODEL_PATH):
    error_msg = f"""
    Error: No se encontró el archivo del modelo en {MODEL_PATH}.
    Asegúrate de que el archivo model.pkl existe en el directorio data/
    y que se copia correctamente durante la construcción de la imagen Docker.
    """
    raise RuntimeError(error_msg)

# Cargar el modelo
print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    raise

model = joblib.load(MODEL_PATH)
print("Modelo cargado:", model)

@app.post("/predict")
async def predict(payload: Landmarks):
    arr = np.array(payload.landmarks, dtype=float).reshape(1, -1)
    # Si tu pipeline necesita preprocesamiento, asegúrate de que model incluya ese pipeline
    pred = model.predict(arr)[0]
    return {"prediction": str(pred)}
