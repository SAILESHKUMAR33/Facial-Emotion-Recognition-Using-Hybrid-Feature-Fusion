import cv2
cap = cv2.VideoCapture(0)
print("Camera 0 opened:", cap.isOpened())
cap2 = cv2.VideoCapture(1)
print("Camera 1 opened:", cap2.isOpened())
cap.release()
cap2.release()
