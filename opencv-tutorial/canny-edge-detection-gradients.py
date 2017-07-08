"""
https://pythonprogramming.net/canny-edge-detection-gradients-python-opencv-tutorial/
"""
import cv2

def demo1():
    """ 1
    """
    while True:
        _, frame = cap.read()

        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.Canny(frame, 50, 50)

        cv2.imshow('original', frame)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)
        cv2.imshow('edges', edges)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


cap = cv2.VideoCapture(0)

demo1()

cv2.destroyAllWindows()
cap.release()
