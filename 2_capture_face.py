import cv2
import os

# Automatically create 'dataset' folder if it doesn't exist (Much better & easier)
if not os.path.exists('dataset'):
    os.makedirs('dataset')

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Input User ID (Use numbers: 1, 2, 3, etc.)
user_id = input('Enter User ID (Number): ')
count = 0

print("Look at the camera. Capturing face samples...")

while True:
    success, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        # Save the cropped face image into dataset folder
        # Format filename exactly like Medium: User.id.count.jpg
        cv2.imwrite("dataset/User." + str(user_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
        
        # Draw red rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow("Step 2 - Face Data Collection", frame)
    cv2.waitKey(1)
    
    # Automatically stop after 25 samples (As specified in Medium Part 2)
    if count >= 25:
        break

print(f"Successfully collected {count} samples for User ID {user_id}!")
video.release()
cv2.destroyAllWindows()