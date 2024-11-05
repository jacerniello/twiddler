from twiddler.helper import cubic_bezier_curve
from twiddler.shape import Shape
import numpy as np
from twiddler.helper import *


class Path(Shape):
    def __init__(self, nested_points, nested_starts, **style_kwargs):
        """for shapes with cursors that can jump around"""
        super().__init__(**style_kwargs)
        self.style_kwargs = style_kwargs
        if not isinstance(nested_points[0], np.ndarray):
            self.points = [np.array(x, dtype='float64') for x in nested_points if x]
        else:
            self.points = nested_points

        if not isinstance(nested_starts[0], np.ndarray):
            self.start = [np.array(s, dtype='float64') for s in nested_starts if s]
        else:
            self.start = nested_starts
        self.close_paths = []

        for i, point_g in enumerate(self.points):
            if (abs(self.start[i] - point_g[-1]) < np.array([0.1, 0.1], dtype='float64')).all():  # if the first and last points are close enough to each other
                self.close_paths.append(True)
            else:
                self.close_paths.append(False)

    def close(self, close=True):
        self.close_paths = [close] * len(self.close_paths)

    @property
    def flattened_points(self):
        a = []
        for x in self.points:
            a.extend(x)
        return a

    def copy(self):
        return Path(self.points, self.start, **self.style_kwargs)

    def move(self, m_d):
        m_x, m_y = m_d
        for i in range(len(self.points)):
            self.points[i][:, 0] += m_x
            self.points[i][:, 1] += m_y

        self.start += np.array(m_d)

    def move_to(self, new_point, center=True):
        if center is True:
            old_point = self.center
        else:
            old_point = self.start[0]
        change = np.array(new_point) - np.array(old_point)
        self.move(change)

    def scale(self, x_s, y_s):
        if x_s != 1 or y_s != 1:
            if x_s > 0:
                x_s_i = 1
            else:
                x_s_i = -1

            if y_s > 0:
                y_s_i = 1
            else:
                y_s_i = -1

            for i in range(len(self.points)):
                start = self.anchor
                t_x, t_y = (start[0] * (x_s_i - x_s), start[1] * (y_s_i - y_s))  # determines how much to translate by
                self.points[i][:, 0] *= x_s
                self.points[i][:, 0] += t_x
                self.points[i][:, 1] *= y_s
                self.points[i][:, 1] += t_y

                self.start[i][0] *= x_s
                self.start[i][0] += t_x
                self.start[i][1] *= y_s
                self.start[i][1] += t_y

            if self.stroke_width:
                self.stroke_width *= max(x_s, y_s)

    def rotate(self, about_point, angle=0, clockwise=True, in_rad=False):
        angle = angle if clockwise else -angle
        angle = format_angle(angle) if not in_rad else angle
        for p in range(len(self.points)):
            distance = self.points[p] - about_point
            x_cos, y_sin, x_sin, y_cos = distance[:, 0] * math.cos(angle), distance[:, 1] * math.sin(angle), distance[:,
                                                                                                             0] * math.sin(
                angle), distance[:, 1] * math.cos(angle)

            self.points[p][:, 0] = x_cos - y_sin + about_point[0]
            self.points[p][:, 1] = x_sin + y_cos + about_point[1]

        for s in range(len(self.start)):
            start = self.start[s] - about_point
            self.start[s] = start[0] * math.cos(angle) - start[1] * math.sin(angle) + about_point[0], \
                         start[0] * math.sin(angle) + start[1] * math.cos(angle) + about_point[1]
