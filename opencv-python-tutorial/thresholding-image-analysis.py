"""
https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/
"""

import cv2

def demo1():
    """ 1
    """
    _, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
    gaus = cv2.adaptiveThreshold(
        grayscaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        115, 1)
    _, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)

    cv2.imshow('original', img)
    cv2.imshow('threshold', threshold)
    cv2.imshow('threshold2', threshold2)
    cv2.imshow('gaus', gaus)
    cv2.imshow('otsu', otsu)

img = cv2.imread('bookpage.jpg')

demo1()

cv2.waitKey(0)
cv2.destroyAllWindows()
