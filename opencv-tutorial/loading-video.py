"""
https://pythonprogramming.net/loading-video-python-opencv-tutorial/?completed=/loading-images-python-opencv-tutorial/
"""
import cv2
from matplotlib import pyplot as plt


def demo1():
    """ demo1
    """
    cap = cv2.VideoCapture(0)

    # plt.axis([0, 10, 0, 1])
    plt.ion()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(frame.shape)
        # plt.scatter(i, y)
        plt.imshow(frame)
        plt.show()
        plt.pause(0.05)

    cap.release()


def demo2():
    """ demo 2
    """
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        cv2.imshow('frame', gray)
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


demo1()
