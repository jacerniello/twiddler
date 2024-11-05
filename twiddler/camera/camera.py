import numpy as np
from twiddler.animate.animate import CameraAnimate


class Camera:
    """defines a Camera object; allows for moving around on the screen"""
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.size = np.array([width, height], dtype='float64')
        self.saved = False
        self.animate = CameraAnimate(self)

    @property
    def center(self):
        return self.size / 2

    def scale_to(self, start, new_size):
        scale = self.size / new_size
        self.canvas.scale(*scale)
        translate = -start
        self.canvas.translate(*translate)

    def save(self):
        self.canvas.save()
        self.saved = True

    def restore(self):
        if self.saved:
            self.canvas.restore()
            self.saved = False

