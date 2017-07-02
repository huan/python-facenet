"""
https://pythonprogramming.net/color-filter-python-opencv-tutorial/
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

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


cap = cv2.VideoCapture(0)

demo1()

cv2.destroyAllWindows()
cap.release()
