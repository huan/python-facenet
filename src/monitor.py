"""
dialog module
"""
from time import sleep
from typing import List, overload, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]
Boxes = List[Box]

class Monitor(object):
    """
    dialog
    """

    def __init__(self, name: str):
        plt.ion()

        self.fig = plt.figure(name, figsize=(3, 3))
        self.ax = self.fig.add_axes([0, 0, 1, 1])

        self.ax.axis('off')
        self.fig.show()

    def display(self, fileOrArray: Union[str, np.ndarray]) -> None:
        """ show photo
        """
        self.ax.clear()

        if isinstance(fileOrArray, str):
            img = plt.imread(fileOrArray)
        else:
            img = fileOrArray

        self.fig.figimage(img, resize=True)     # how to resize instead of this?
        self.ax.imshow(img)

    # # pylint: disable=E0102

    # @overload
    # def rectangle(self, boxes: np.ndarray) -> None:
    #     """doc"""
    #     pass

    def boxes(self, boxes: np.ndarray) -> None:
        """draw boxes"""
        int_boxes = boxes.astype(np.int32)
        for i in range(int_boxes.shape[0]):
            m_x = int_boxes[i][0]
            m_y = int_boxes[i][1]
            m_w = int_boxes[i][2] - m_x
            m_h = int_boxes[i][3] - m_y
            self.rectangle(m_x, m_y, m_w, m_h)

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
            linewidth=1,
            edgecolor=np.random.rand(3),
            # facecolor=np.random.rand(3),
            facecolor='none',
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
        # self.fig.waitforbuttonpress()
        plt.waitforbuttonpress()

    def plot(self, *args, **kwargs) -> None:
        """plot proxy"""
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.plot(*args, **kwargs)
