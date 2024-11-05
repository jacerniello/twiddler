import numpy as np
from twiddler.animate.shape_animate import ShapeAnimation
from twiddler.helper import array_round


class GradientAnimation(ShapeAnimation):
    def __init__(self, animate, parent, kwargs):
        super().__init__(animate, parent, kwargs)

    def move_gradient_to(self, target_positions, change=None, **kwargs):

        if change is None:
            change = (np.array(target_positions, dtype='float64') - np.array(self.parent.coordinates, dtype='float64')) * self.alpha
        if self.animation_start_pos:
            self.parent.coordinates += change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.coordinates += change

        self.kwargs = {'target_positions': target_positions, 'change': change}

    def transform_gradient(self, g_new, change=None, **kwargs):
        """converts gradient"""
        if change is None:
            change = []
            while len(g_new.colors) != len(self.parent.colors):
                if len(g_new.colors) > len(self.parent.colors):
                    self.parent.colors.append([[0, 0, 0, 0], 1])
                else:
                    g_new.colors.append([[0, 0, 0, 0], 1])
            for i in range(len(self.parent.colors)):
                c = []
                for j in range(2):
                    c.append((np.array(g_new.colors[i][j], dtype='float64') - np.array(self.parent.colors[i][j], dtype='float64')) * self.alpha)
                change.append(c)

        if self.animation_start_pos:
            for i, x in enumerate(change):
                for j in range(2):
                    self.parent.colors[i][j] += change[i][j] * self.animation_start_pos
                    self.parent.colors[i][j] = array_round(self.parent.colors[i][j], 100)  # prevent underflow issue

            self.animation_start_pos = 0
        else:
            for i, x in enumerate(change):
                for j in range(2):
                    self.parent.colors[i][j] += change[i][j]
                    self.parent.colors[i][j] = array_round(self.parent.colors[i][j], 100)  # prevent underflow issue
            # print('a', self.parent.colors, 'b', g_new.colors)

        self.kwargs = {'g_new': g_new, 'change': change}
