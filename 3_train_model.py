import cv2
import os
import numpy as np
from PIL import Image

# Automatically create 'models' folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images_with_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        # Open image and convert to grayscale using PILLOW
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        
        # Extract ID from filename (e.g., dataset/User.1.12.jpg -> takes the number 1)
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y+h, x:x+w])
            ids.append(user_id)
            
    return face_samples, ids

print("Training the AI model... Please wait.")
faces, ids = get_images_with_labels('dataset')

# Train the recognizer with face samples and their corresponding IDs
recognizer.train(faces, np.array(ids))

# Save the trained model to models folder
recognizer.save('models/trained_model.xml')
print("Training completed! Model saved as 'models/trained_model.xml'")