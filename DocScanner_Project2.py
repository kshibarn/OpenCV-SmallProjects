import cv2
import numpy as np

widthImg = 480
heightImg = 640

cap = cv2.VideoCapture(1)
cap.set(3, 480)
cap.set(4, 640)
cap.set(10, 150)


def preProcessing(img):  # Different PreProcessed Image using different basic functions of OpenCV
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (4, 4), 2)
    imgCanny = cv2.Canny(imgBlur, 250, 250)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(Points):
    Points = Points.reshape((4, 2))
    Points_new = np.zeroes((4, 1, 2), np.int32)
    add = Points.sum(1)
    Points_new[0] = Points[np.argmin(add)]
    Points_new[3] = Points[np.argmax(add)]
    diff = np.diff(Points, axis=1)
    Points_new[1] = Points[np.argmin(diff)]
    Points_new[2] = Points[np.argmax(diff)]

    return Points_new


def getWarp(img, biggest):  # To get selected Document part
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20,
                 20:imgOutput.shape[1] - 20]  # Cropped Image to get a more cleaner image
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)

    cv2.imshow("Result", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
