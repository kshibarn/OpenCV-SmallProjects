import cv2

# Read Images
img = cv2.imread("Resources/JL.png")

cv2.imshow("Output", img)
cv2.waitKey(0)

# Read Video
cap = cv2.VideoCapture("Resources/Sample-Video.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

# Read Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
