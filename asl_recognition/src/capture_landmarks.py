import cv2
import mediapipe as mp
import csv
import os

# Rutas
DATA_FILE = "data/landmarks.csv"
os.makedirs("data", exist_ok=True)

# Configurar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Si el archivo no existe, crear encabezado
if not os.path.isfile(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["letra"] + [f"lm_{i}" for i in range(63)]
        writer.writerow(header)

# Función para capturar muestras de una letra
def capture_samples(letra):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print(f"Mostrando la letra {letra}. Captura las muestras y presiona 'q' para terminar.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                # Guardar en CSV
                with open(DATA_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([letra] + coords)
                count += 1
                print(f"Muestra {count} para {letra}")

        cv2.putText(frame, f"Letra: {letra} | Muestras: {count} | 'q' para salir",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Captura de Señales", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura de {letra} completada. Total: {count} muestras\n")

# Capturar todas las letras A-Z
for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    input(f"Presiona Enter para comenzar con {letra}...")
    capture_samples(letra)
