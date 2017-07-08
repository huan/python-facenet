"""
dialog module
"""
from time import sleep
from typing import Tuple

import matplotlib.pyplot as plt
# from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle

import numpy as np

Point = Tuple[int, int]

i = 3   # type: int
i = '3'


class Ui(object):
    """
    dialog
    """

    def __init__(self):
        self.im = None   # type: AxesImage
        self.fig, self.ax = plt.subplots()
        self.interactive_mode()
        self.show()

    def show_photo(
            self,
            file: str
    ) -> None:
        """ show photo
        """
        img = plt.imread(file)
        self.im = plt.imshow(img)

    def draw_rectangle(
            self,
            x: int,
            y: int,
            width: int,
            height: int,
    ) -> None:
        """ draw """
        rgba = np.random.rand(4,)
        # color = color.list()
        rect = Rectangle(
            [x, y],
            width,
            height,
            linewidth=1,
            color=rgba,
            edgecolor='r',
            facecolor='none'
        )
        self.ax.add_patch(rect)

    def draw(self) -> None:
        """ refresh plt show """
        # self.ax.draw()
        plt.draw()

    def show(self):
        """ show """
        self.ax.axis('off')  # clear x- and y-axes
        plt.show()

    def pause(self, seconds: int) -> None:
        """ pause """
        plt.pause(seconds)

    def sleep(self, seconds: int) -> None:
        """ sleep """
        sleep(seconds)

    def interactive_mode(self, switch: bool = True) -> None:
        """ interactive """
        if switch:
            plt.ion()
        else:
            plt.ioff()

    def wait_key(self) -> None:
        """ wait press """
        # plt.pause(1) # <-------
        # raw_input("<Hit Enter To Close>")
        plt.waitforbuttonpress()

    def huan(self, n: int) -> None:
        """ test
        """
        print('huan%d' % (n))

