# -*- coding: utf-8 -*-
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
import threading
import pygame
import time
import os
from utils import save_visitor, timestamp_now

# Load trained model
def load_model(filename='face_recognition_model.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['threshold']

# Init pygame mixer
pygame.mixer.init()
sound_channel = pygame.mixer.Channel(0)
known_sound = pygame.mixer.Sound("sounds/known_bell.mp3")
unknown_sound = pygame.mixer.Sound("sounds/unknown_bell.mp3")

# Load Models
print("Loading models...")
model, encoder, threshold = load_model()
embedder = FaceNet()
detector = MTCNN()

# Preprocessing
def preprocess_face(face_img, target_size=(160, 160)):
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img.astype('float32')
    return face_img

# Embedding
def get_embedding(face_img):
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

# Prediction
def predict_identity(face_img):
    embedding = get_embedding(face_img)
    prob = model.predict_proba([embedding])[0]
    max_prob = np.max(prob)
    if max_prob < threshold:
        return "Unknown", max_prob
    else:
        pred_class = model.predict([embedding])[0]
        return encoder.inverse_transform([pred_class])[0], max_prob

# Sound
def play_bell(identity):
    if sound_channel.get_busy():
        sound_channel.stop()

    if identity == "Unknown":
        sound_channel.play(unknown_sound)
    else:
        sound_channel.play(known_sound)

# Recognition Process
def recognize_person(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    if results:
        result = results[0]
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        face = rgb_frame[y:y+h, x:x+w]
        face = preprocess_face(face)
        identity, confidence = predict_identity(face)

        print(f"[{timestamp_now()}] {identity} ({confidence:.2f})")
        play_bell(identity)
        save_visitor(identity, frame)

    else:
        print(f"[{timestamp_now()}] No face detected.")

# Camera Initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("SMART DOORBELL READY 🚪🔔 Press 'b' to detect, 'q' to quit.")

while True:
    key = input("Press key: ").strip().lower()

    if key == 'b':
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue

        threading.Thread(target=recognize_person, args=(frame,)).start()

    elif key == 'q':
        break

cap.release()
pygame.mixer.quit()
