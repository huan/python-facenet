"""
https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/
"""
import cv2
import numpy as np

def demo1():
    """ 1
    """
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # hsv hue sat value
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])
        # do not use this
        # dark_red = np.uint8([[[12,22,1221]]])
        # dark_red = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dialation = cv2.dilate(mask, kernel, iterations=1)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('frame', frame)
        cv2.imshow('res', res)
        cv2.imshow('erosion', erosion)
        cv2.imshow('dialation', dialation)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


cap = cv2.VideoCapture(0)

demo1()

cv2.destroyAllWindows()
cap.release()
