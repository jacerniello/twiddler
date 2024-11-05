from twiddler.animate.shape_animate import ShapeAnimation
from twiddler.helper import transform_fix_shapes
from twiddler.color.color import format_color
import numpy as np


class GroupAnimation(ShapeAnimation):
    def __init__(self, animate, parent, kwargs):
        super().__init__(animate, parent, kwargs)

    def move(self, m_d, center=True, change=None, animation_target=None, **kwargs):
        if animation_target is None:
            if center or self.parent.animate_at_center:
                animation_target = self.parent.center
            elif self.parent.anchor is not None:
                animation_target = self.parent.anchor
            else:
                animation_target = self.parent.start

        if change is None:
            change = np.array(m_d, dtype='float64') * self.alpha
        if self.animation_start_pos:
            for elem in self.parent.flattened_elements:
                elem.start += change * self.animation_start_pos
                elem.anchor += change * self.animation_start_pos
                for p in range(len(elem.points)):
                    elem.points[p] += change * self.animation_start_pos

            self.animation_start_pos = 0
        else:
            for elem in self.parent.flattened_elements:
                elem.start += change
                elem.anchor += change
                for p in range(len(elem.points)):
                    elem.points[p] += change

        self.kwargs = {'m_d': m_d, 'center': center, 'change': change, 'animation_target': animation_target}

    def move_to(self, point, center=True, change=None, animation_target=None, **kwargs):
        if animation_target is None:
            if center or self.parent.animate_at_center:
                animation_target = self.parent.center
            elif self.parent.anchor is not None:
                animation_target = self.parent.anchor
            else:
                animation_target = self.parent.start

        if change is None:
            change = (point - animation_target) * self.alpha

        if self.animation_start_pos:
            for elem in self.parent.flattened_elements:
                elem.start += change * self.animation_start_pos
                elem.anchor += change * self.animation_start_pos
                for p in range(len(elem.points)):
                    elem.points[p] += change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            for elem in self.parent.flattened_elements:
                elem.start += change
                elem.anchor += change
                for p in range(len(elem.points)):
                    elem.points[p] += change

        self.kwargs = {'point': point, 'center': center, 'change': change, 'animation_target': animation_target}

    def transform(self, new_shape, start_change=None, point_change=None, color_change=None, same_size=False, **kwargs):
        if same_size is False:
            transform_fix_shapes(new_shape, self.parent)
            same_size = True
        if start_change is None:
            start_change = (new_shape.start - self.parent.start) * self.alpha
        if point_change is None:
            point_change = (new_shape.points - self.parent.points) * self.alpha
        if color_change is None:
            color_change = (new_shape.color - self.parent.color) * self.alpha

        if self.animation_start_pos:
            self.parent.start += start_change * self.animation_start_pos
            self.parent.points += point_change * self.animation_start_pos
            self.parent.color += color_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.start += start_change
            self.parent.points += point_change
            self.parent.color += color_change
        self.kwargs = {'new_shape': new_shape, 'start_change': start_change, 'point_change': point_change, 'color_change': color_change, 'same_size': same_size}

    def color(self, new_color, color_change=None, **kwargs):
        if color_change is None:
            new_color = format_color(new_color)
            color_change = (new_color - self.parent.color) * self.alpha

        if self.animation_start_pos:
            self.parent.color += color_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.color += color_change

        self.kwargs = {'new_color': new_color, 'color_change': color_change}

    def bool_op(self, **kwargs):
        self.parent.remove(-1)  # removes all old shapes so they can be redrawn
        self.parent.multi_shapes(self.parent.shapes)
        self.kwargs = {}

    def union(self, **kwargs):
        self.parent.remove(-1)  # removes all old shapes so they can be redrawn
        self.parent.make_union()
        self.kwargs = {}

    def rotate(self, about_point, angle, repeat, clockwise, angle_change=None, **kwargs):

        if angle_change is None:
            angle *= repeat
            angle_change = angle * self.alpha

        if self.animation_start_pos:
            for elem in self.parent.flattened_elements:
                elem.rotate(about_point, angle=angle_change * self.animation_start_pos, clockwise=clockwise)
            self.animation_start_pos = 0
        else:
            for elem in self.parent.flattened_elements:
                elem.rotate(about_point, angle=angle_change, clockwise=clockwise)

        self.kwargs = {'about_point': about_point, 'angle': angle, 'repeat': repeat,
                       'clockwise': clockwise, 'angle_change': angle_change}

    def create(self, alpha_total=0, **kwargs):
        if self.animation_start_pos:
            alpha_total = self.alpha * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            alpha_total += self.alpha

        for elem in self.parent.flattened_elements:
            elem.create(alpha_total)
        self.kwargs = {'alpha_total': alpha_total}
