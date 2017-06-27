"""
https://pythonprogramming.net/loading-images-python-opencv-tutorial/
"""

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray', interpolation='bicubic')
# to hide tick values on X and Y axis
plt.xticks([])
plt.yticks([])
plt.plot([200, 300, 400], [100, 200, 300], 'c', linewidth=5)
plt.show()

cv2.imwrite('imagegray.png', img)
