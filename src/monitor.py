"""
dialog module
"""
from time import sleep
from typing import Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

Point = Tuple[int, int]


class Monitor(object):
    """
    dialog
    """

    def __init__(self, name: str):
        plt.ion()

        self.fig = plt.figure(name)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

        self.ax.axis('off')
        self.fig.show()

    def display(self, fileOrArray: Union[str, np.ndarray]) -> None:
        """ show photo
        """
        if isinstance(fileOrArray, str):
            img = plt.imread(fileOrArray)
        else:
            img = fileOrArray

        # self.fig.figimage(img, resize=True)
        self.ax.imshow(img)

    def rectangle(
            self,
            x: int,
            y: int,
            width: int,
            height: int,
    ) -> None:
        """ draw """
        rect = Rectangle(
            [x, y],
            width,
            height,
            linewidth=0.5,
            edgecolor=np.random.rand(3),
            facecolor='none'
        )
        self.ax.add_patch(rect)

    def draw(self) -> None:
        """ refresh plt show """
        # self.ax.draw()
        plt.draw()

    # def pause(self, seconds: int) -> None:
    #     """ pause """
    #     plt.pause(seconds)

    def sleep(self, seconds: int) -> None:
        """ sleep """
        sleep(seconds)

    def waitforbuttonpress(self) -> None:
        """ wait for any key """
        self.fig.waitforbuttonpress()
