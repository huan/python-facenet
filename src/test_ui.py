""" module """
from ui import Ui

ui = Ui()

ui.show_photo('../opencv-python-tutorial/brothers.jpg')
ui.draw_rectangle(5, 5, 10, 10)

# ui.pause(1)
ui.wait_key()

ui.draw_rectangle(50, 50, 100, 100)
# ui.draw()

# ui.pause(10)
ui.wait_key()
