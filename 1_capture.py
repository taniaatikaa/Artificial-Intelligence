import cv2

# Initialize camera
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

print("Camera is opening... Press 'q' to exit.")

while True:
    success, frame = video.read()
    if not success:
        print("Failed to grab frame")
        break
        
    # Convert frame to grayscale (as explained in Medium Part 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the camera feed
    cv2.imshow("Step 1 - Camera Capture", gray)
    
    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Camera closed.")