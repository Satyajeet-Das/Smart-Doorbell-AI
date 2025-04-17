# -*- coding: utf-8 -*-
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# Load your trained model
def load_model(filename='face_recognition_model.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['threshold']

# Initialize models
print("Loading models...")
model, encoder, threshold = load_model()
embedder = FaceNet()
detector = MTCNN()

# Face preprocessing
def preprocess_face(face_img, target_size=(160, 160)):
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img.astype('float32')
    return face_img

# Get embedding
def get_embedding(face_img):
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

# Prediction function
def predict_identity(face_img):
    embedding = get_embedding(face_img)
    prob = model.predict_proba([embedding])[0]
    max_prob = np.max(prob)
    
    if max_prob < threshold:
        return "unknown", max_prob
    else:
        pred_class = model.predict([embedding])[0]
        return encoder.inverse_transform([pred_class])[0], max_prob

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting face recognition. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert to RGB (MTCNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = detector.detect_faces(rgb_frame)
    
    # Process each face
    for result in results:
        x, y, w, h = result['box']
        
        # Adjust coordinates if they go out of frame
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        # Extract face
        face = rgb_frame[y:y+h, x:x+w]
        face = preprocess_face(face)
        
        # Predict identity
        identity, confidence = predict_identity(face)
        
        # Draw rectangle and label
        color = (0, 255, 0) if identity != "unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{identity} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()