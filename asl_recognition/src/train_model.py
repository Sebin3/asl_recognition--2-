import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Cargar dataset
df = pd.read_csv("data/landmarks.csv")
X = df.drop("letra", axis=1)
y = df["letra"]

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluar precisión
score = model.score(X_test, y_test)
print(f"Precisión del modelo: {score*100:.2f}%")

# Guardar modelo
with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo guardado en data/model.pkl")
