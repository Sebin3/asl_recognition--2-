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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # solo para desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "data/model.pkl"
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. Entrena y guarda model.pkl en data/")

model = joblib.load(MODEL_PATH)
print("Modelo cargado:", model)

@app.get("/")
def read_root():
    return {"message": "API de reconocimiento de señas activa. Usa el endpoint /predict para predecir una letra."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(payload: Landmarks):
    arr = np.array(payload.landmarks, dtype=float).reshape(1, -1)
    # Si tu pipeline necesita preprocesamiento, asegúrate de que model incluya ese pipeline
    pred = model.predict(arr)[0]
    return {"prediction": str(pred)}# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

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

MODEL_PATH = "data/model.pkl"
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
