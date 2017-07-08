""" module """
from monitor import Monitor

m1 = Monitor('1')
m1.display('../opencv-python-tutorial/brothers.jpg')

m1.rectangle(5, 5, 10, 10)
m1.waitforbuttonpress()

m1.display('../opencv-python-tutorial/tux-python-egg.jpg')
m1.waitforbuttonpress()
