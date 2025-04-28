# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:10:29 2025

@author: Steven
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque
from spellchecker import SpellChecker  # Importamos el corrector ortográfico

# ======== Cargar modelo y codificador de etiquetas ========
modelo = load_model("modelo_lsc_mediapipe.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ======== Inicializar MediaPipe ========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# ======== Inicializar corrector ortográfico ========
spell = SpellChecker(language='es')

# Función para corregir palabra y obtener varias alternativas
# Función para corregir palabra y permitir elección
def corregir_palabra(palabra):
    palabra = palabra.lower()
    palabras = palabra.split()
    palabras_corregidas = []
    for p in palabras:
        correcciones = list(spell.candidates(p))
        if len(correcciones) > 1:
            print(f"\nPalabra detectada: '{p}'. Alternativas:")
            for idx, alt in enumerate(correcciones):
                print(f"{idx + 1}: {alt}")
            try:
                eleccion = int(input("Selecciona la alternativa correcta (número): ")) - 1
                if 0 <= eleccion < len(correcciones):
                    palabras_corregidas.append(correcciones[eleccion])
                else:
                    palabras_corregidas.append(spell.correction(p))
            except ValueError:
                palabras_corregidas.append(spell.correction(p))
        else:
            palabras_corregidas.append(spell.correction(p))
    return ' '.join(palabras_corregidas)

# ======== Iniciar cámara ========
cap = cv2.VideoCapture(0)
print("Iniciando cámara... Presiona 'q' para salir.")

# Variables para almacenar la palabra formada
palabra_actual = ""
letra_anterior = ""
tiempo_sin_letra = 0  # Contador para tiempo sin letras
historial_palabras = []  # Guardar las palabras detectadas

# Variables para suavizar la predicción
ventana_predicciones = deque(maxlen=5)  # Promediar las últimas 5 predicciones

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letra_actual = None  # Inicializar

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
                landmark_list.append(lm.z)

            landmark_array = np.array(landmark_list).reshape(1, -1)

            # Realizar predicción
            pred = modelo.predict(landmark_array, verbose=0)
            pred_idx = np.argmax(pred)
            letra_predicha = le.inverse_transform([pred_idx])[0]
            confianza = np.max(pred)

            # Guardar predicción en la ventana
            if confianza > 0.8:
                ventana_predicciones.append(letra_predicha)

            # Solo cuando tengamos suficientes predicciones suavizamos
            if len(ventana_predicciones) == ventana_predicciones.maxlen:
                letra_suavizada = max(set(ventana_predicciones), key=ventana_predicciones.count)
                letra_actual = letra_suavizada

                # Agregar letra si cambia
                if letra_actual != letra_anterior:
                    palabra_actual += letra_actual
                    letra_anterior = letra_actual
                    tiempo_sin_letra = 0
            else:
                tiempo_sin_letra += 1

    else:
        tiempo_sin_letra += 1

    # Si hay inactividad (ejemplo 30 frames) terminamos palabra
    if tiempo_sin_letra > 30 and palabra_actual:
        # Autocorregir la palabra usando pyspellchecker
        palabra_correccion = corregir_palabra(palabra_actual)
        historial_palabras.append(palabra_correccion)
        print(f"Palabra detectada: {palabra_actual} -> Corregida: {palabra_correccion}")
        palabra_actual = ""
        letra_anterior = ""
        ventana_predicciones.clear()
        tiempo_sin_letra = 0

    # Mostrar letra suavizada si existe
    if len(ventana_predicciones) == ventana_predicciones.maxlen:
        cv2.putText(frame, f"{letra_actual} ", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Detectando...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Mostrar palabra en formación
    cv2.putText(frame, f"Palabra: {palabra_actual}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Mostrar historial de palabras
    texto_historial = " ".join(historial_palabras[-5:])  # Solo mostrar las últimas 5 palabras
    cv2.putText(frame, f"Historial: {texto_historial}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Reconocimiento LSC", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




