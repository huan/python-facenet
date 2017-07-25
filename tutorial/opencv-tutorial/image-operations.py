"""
https://pythonprogramming.net/image-operations-python-opencv-tutorial/
"""
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
px = img[55, 55]

img[55, 55] = [255, 255, 255]
px = img[55, 55]
print(px)

px = img[100:150, 100:150]
print(px)

img[100:150, 100:150] = [255, 255, 255]
print(img.shape)
print(img.size)
print(img.dtype)

watch_face = img[37:111, 107:194]
img[0:74, 0:87] = watch_face

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
