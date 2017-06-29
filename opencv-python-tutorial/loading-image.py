"""
https://pythonprogramming.net/loading-images-python-opencv-tutorial/
"""

from matplotlib import pyplot as plt


img = plt.imread('brothers.jpg')

plt.imshow(img, cmap='gray', interpolation='bicubic')
# to hide tick values on X and Y axis
# plt.xticks([])
# plt.yticks([])
plt.axis("off")

plt.plot([200, 300, 400], [100, 200, 300], 'c', linewidth=5)
plt.show()

# cv2.imwrite('imagegray.png', img)
