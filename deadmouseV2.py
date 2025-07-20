import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from openvino.inference_engine import IECore
import os
import time
import pygame

# --- Inizializzazione di Pygame ---
pygame.init()

# --- Configurazione Pygame ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Muovi Pallino con Indice")

# Colori (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Dati del Quadrato Esterno
SQUARE_RECT = pygame.Rect(150, 100, 500, 400)  # x, y, larghezza, altezza
SQUARE_COLOR = BLACK
SQUARE_BORDER_THICKNESS = 3  # Spessore del bordo

# Dati del Pallino
ball_radius = 20  # Raggio iniziale
ball_color = RED
ball_x = SQUARE_RECT.centerx  # Posizione iniziale al centro del quadrato
ball_y = SQUARE_RECT.centery
step_size = 5  # Incremento di movimento per direzione

# --- Configurazione AI e Webcam ---
device = "MYRIAD"  # Usa MYRIAD se disponibile, altrimenti CPU
size = 256
keypoint_history_length = 16
threshold_movement = 0.02  # Soglia ridotta per rilevare movimenti lenti
direction_stability_threshold = 10  # Aumentato per maggiore stabilità con movimenti lenti
frame_delay = 50  # Ritardo di 50ms (20 FPS) per rallentare la lettura

# Modelli
model_dir = "ir_models"
tflite_dir = "tflite"  # Directory per il modello TFLite
tflite_model = f"{tflite_dir}/hand_landmark.tflite"
print("Percorso del modello:", tflite_model)  # Debug del percorso

keypoint_xml = f"{model_dir}/keypoint_classifier/keypoint_classifier.xml"
keypoint_bin = keypoint_xml.replace(".xml", ".bin")
history_xml = f"{model_dir}/point_history_classifier/point_history_classifier_FINAL_MYRIAD_COMPAT.xml"
history_bin = history_xml.replace(".xml", ".bin")

# Verifica file modelli
if not os.path.exists(tflite_model):
    print(f"Errore: {tflite_model} non trovato.")
    exit()
if not os.path.exists(keypoint_xml) or not os.path.exists(keypoint_bin):
    print(f"Errore: file OpenVINO per keypoint_classifier non trovati.")
    exit()
if not os.path.exists(history_xml) or not os.path.exists(history_bin):
    print(f"Errore: file OpenVINO per point_history_classifier non trovati.")
    exit()

# TensorFlow Lite
tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model)
tflite_interpreter.allocate_tensors()
tflite_input = tflite_interpreter.get_input_details()[0]
tflite_output = tflite_interpreter.get_output_details()[0]
h = w = 256

# OpenVINO
ie = IECore()
try:
    k_net = ie.read_network(model=keypoint_xml, weights=keypoint_bin)
    k_exec = ie.load_network(network=k_net, device_name=device)
    k_input = next(iter(k_net.inputs))
    k_output = next(iter(k_net.outputs))
except Exception as e:
    print(f"Errore nel caricamento del modello keypoint_classifier su {device}: {e}")
    exit()

try:
    h_net = ie.read_network(model=history_xml, weights=history_bin)
    h_exec = ie.load_network(network=h_net, device_name=device)
    h_input = next(iter(h_net.inputs))
    h_output = next(iter(h_net.outputs))
except Exception as e:
    print(f"Errore nel caricamento del modello point_history_classifier su {device}: {e}")
    exit()

print("Numero di classi keypoint:", k_net.outputs[k_output].shape[1])
print("Numero di classi history:", h_net.outputs[h_output].shape[1])

# History e stabilizzazione
point_history = deque(maxlen=keypoint_history_length)
last_direction = "-"
direction_counter = 0  # Conta i frame consecutivi con la stessa direzione

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Errore: impossibile aprire la telecamera.")
    exit()

print("🎥 Avvio inferenza — premi ESC per uscire")

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Errore: frame non catturato.")
        break

    fh, fw = frame.shape[:2]
    print("Dimensioni frame:", fw, fh)
    cx, cy = fw // 2, fh // 2
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    roi = frame[y1:y2, x1:x2]

    if roi.shape[:2] != (size, size):
        print("Errore: ROI non valida, dimensioni:", roi.shape)
        continue

    # Migliora contrasto
    roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)

    cv2.imshow("ROI", roi)
    cv2.waitKey(1)  # Forza l'aggiornamento della finestra ROI

    img = cv2.resize(roi, (w, h)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    tflite_interpreter.set_tensor(tflite_input['index'], img)
    tflite_interpreter.invoke()
    landmarks = tflite_interpreter.get_tensor(tflite_output['index'])[0]  # Corretto a tflite_interpreter e tflite_output
    print("Landmarks:", landmarks)

    if np.all(landmarks == 0):
        print("Nessun landmark rilevato.")
        continue

    # Estrai solo il keypoint 8 (punta dell'indice)
    index_x = landmarks[8 * 3 + 0]  # Coordinata x del keypoint 8
    index_y = landmarks[8 * 3 + 1]  # Coordinata y del keypoint 8
    px = index_x
    py = index_y

    # Disegna una cornice verde intorno all'indice
    frame_size = 20  # Dimensione della cornice (20x20 pixel)
    top_left_x = int((index_x * size) - frame_size // 2)
    top_left_y = int((index_y * size) - frame_size // 2)
    bottom_right_x = int((index_x * size) + frame_size // 2)
    bottom_right_y = int((index_y * size) + frame_size // 2)
    cv2.rectangle(roi, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), GREEN, 2)

    cv2.circle(roi, (int(index_x * size), int(index_y * size)), 3, (0, 255, 0), -1)  # Visualizza il keypoint 8

    point_history.append((px, py))

    direction = "-"
    if len(point_history) == keypoint_history_length:
        start = np.mean([p for p in point_history], axis=0)  # Media mobile per ridurre il jitter
        end = point_history[-1]
        dist = np.linalg.norm(np.array(end) - np.array(start))
        print("Distanza movimento:", dist)

        if dist >= threshold_movement:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if abs(dx) > abs(dy):
                if dx > 0.02:  # Soglia ridotta per movimenti lenti
                    direction = "👉 DESTRA"
                elif dx < -0.02:  # Soglia ridotta per movimenti lenti
                    direction = "👈 SINISTRA"
            else:
                if dy < -0.02:  # Soglia ridotta per movimenti lenti
                    direction = "👆 ALTO"
                elif dy > 0.02:  # Soglia ridotta per movimenti lenti
                    direction = "👇 BASSO"

        if direction != "-" and direction == last_direction:
            direction_counter += 1
        else:
            direction_counter = 1
            last_direction = direction

        if direction_counter >= direction_stability_threshold:
            if direction != "-" and direction != last_direction:
                os.system(f'echo Direzione: {direction}')
                last_direction = direction
            direction_counter = 0  # Resetta dopo aver confermato

    # Aggiorna la posizione della palla in base alla direzione
    if last_direction == "👉 DESTRA" and ball_x < SQUARE_RECT.right - ball_radius:
        ball_x += step_size
    elif last_direction == "👈 SINISTRA" and ball_x > SQUARE_RECT.left + ball_radius:
        ball_x -= step_size
    elif last_direction == "👆 ALTO" and ball_y > SQUARE_RECT.top + ball_radius:
        ball_y -= step_size
    elif last_direction == "👇 BASSO" and ball_y < SQUARE_RECT.bottom - ball_radius:
        ball_y += step_size

    # Disegno
    screen.fill(WHITE)  # Riempie lo sfondo di bianco ad ogni frame
    pygame.draw.rect(screen, SQUARE_COLOR, SQUARE_RECT, SQUARE_BORDER_THICKNESS)  # Disegna il quadrato
    pygame.draw.circle(screen, ball_color, (ball_x, ball_y), ball_radius)  # Disegna il pallino

    # Aggiorna lo Schermo
    pygame.display.flip()

    # Rallenta la lettura
    if cv2.waitKey(frame_delay) & 0xFF == 27:  # Gestione ESC per uscire con ritardo
        running = False

# Uscita
cap.release()
cv2.destroyAllWindows()
pygame.quit()