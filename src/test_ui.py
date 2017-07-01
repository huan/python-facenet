""" module """
from monitor import Monitor

m1 = Monitor('1')
m1.display('../opencv-python-tutorial/brothers.jpg')

m2 = Monitor('2')
m2.display('../opencv-python-tutorial/brothers.jpg')

m1.rectangle(5, 5, 10, 10)
m1.waitforbuttonpress()

m2.rectangle(50, 50, 100, 100)
m2.waitforbuttonpress()
