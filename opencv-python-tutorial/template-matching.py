"""
https://pythonprogramming.net/template-matching-python-opencv-tutorial/
"""
import cv2
import numpy as np

img_bgr = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.75
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    color = np.random.randint(0, 256, 3)
    color = color.tolist()  # https://stackoverflow.com/a/16192589/1123955
    cv2.rectangle(
        img_bgr,
        pt, (pt[0] + w, pt[1] + h),
        color,
        # (0,255,0),
        2)


cv2.imshow('detected', img_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
