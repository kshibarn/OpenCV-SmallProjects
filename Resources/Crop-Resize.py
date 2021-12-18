import cv2

img = cv2.imread("Resources/lambo.png")
print(img.shape)

imgResize = cv2.resize(img, (1200, 500))
print(imgResize.shape)

imgCropped = img[0: 300, 400: 600]

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)
