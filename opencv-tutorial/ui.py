#!/usr/bin/python3.6
"""
ZetCode PyQt5 tutorial

In this example, we create a simple
window in PyQt5.

author: Jan Bodnar
website: zetcode.com
last edited: January 2015
"""

import sys
from typing import Tuple

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

class App(QWidget):
    """ App
    """
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        """ init ui
        """
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        w, h = self.setImage('../test.png')

        self.resize(w, h)
        self.show()

    def setImage(self, file: str):
        """ add image
        """
        # Create widget
        label = QLabel(self)
        pixmap = QPixmap(file)
        label.setPixmap(pixmap)
        return pixmap.width(), pixmap.height()
        # Add paint widget and paint
        # self.m = PaintWidget(self)
        # self.m.move(0, 0)
        # self.m.resize(self.width, self.height)


class PaintWidget(QWidget):
    """ Paint
    """
    def paintEvent(self):  # , event):
        """ paint
        """
        qp = QPainter(self)

        qp.setPen(Qt.black)
        # size = self.size()

        # x = random.randint(1, size.width()-1)
        # y = random.randint(1, size.height()-1)
        # qp.drawPoint(x, y)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
