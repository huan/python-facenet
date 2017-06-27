"""
https://pythonprogramming.net/image-arithmetics-logic-python-opencv-tutorial/?completed=/image-operations-python-opencv-tutorial/
"""

import cv2


def demo1():
    """ 1
    """

    # add = cv2.add(img1, img2)
    # cv2.imshow('add', add)

    weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
    cv2.imshow('weighted', weighted)


def demo2():
    """ 2
    """
    rows, cols, _ = img3.shape
    roi = img1[0:rows, 0:cols]

    img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img3gray, 240, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('mask', mask)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

    dst = cv2.add(img1_bg, img3_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)

img1 = cv2.imread('brothers.jpg')
img2 = cv2.imread('colors.jpg')
img3 = cv2.imread('tux-python-egg.jpg')

# demo1()
demo2()

cv2.waitKey(0)
cv2.destroyAllWindows()
