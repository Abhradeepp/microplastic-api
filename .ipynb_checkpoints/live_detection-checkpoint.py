import cv2 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
else:
    print("Camera detected")
cap.release()
