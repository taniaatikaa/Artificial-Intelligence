import cv2

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('models/trained_model.xml')

font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8

print("Starting Face Recognition... Press 'q' to exit.")

while True:
    success, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Predict the face ID and confidence level
        user_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Calculate matching percentage
        # LBPH confidence returns distance. Lower is better. 0 is perfect match.
        if confidence < 100:
            name = "Tania" 
            confidence_percent = round(100 - confidence)
            display_text = f"{name} {confidence_percent}%"
            color = (0, 255, 0) # Green for recognized
        else:
            name = "Unknown"
            display_text = f"{name}"
            color = (0, 0, 255) 
            
        # Draw rectangle and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y-10), font_face, font_scale, color, 2)
        
    cv2.imshow("Face Recognition Result", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()