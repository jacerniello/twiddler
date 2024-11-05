import numpy as np
import math
from twiddler.constants import *

from twiddler.helper import *
from twiddler.color.color import format_color
from twiddler.animate.animate import Animate, ShapeAnimation
import copy
from twiddler.group import Group


class Shape:
    def __init__(self, points=None, start=(0, 0), color=COLOR_DEFAULT, as_bezier=True, close_path=True, fill=True, stroke=True, stroke_width=0,
                 stroke_color=COLOR_DEFAULT, stroke_opacity=1, anchor=None, fill_opacity=1, line_cap='BUTT', line_join='MITER', fill_rule='NONZERO', **kwargs):
        '''
        :param points: defines the points of the subpaths of a Shape
        :param start: starting position of the Shape
        :param color: color of the shape
        :param close_path: should the path close?
        :param fill: boolean variable, whether to fill the shape
        :param stroke_width:
        :param stroke_color:
        :param anchor: user-defined anchor point of the shape
        '''

        if not as_bezier:
            # convert regular list of points to bezier list of points
            all_points = []
            points.append(points[0])
            for a, b in zip(points, points[1:]):
                all_points.extend(line_to_cubic_bezier(*a, *b))
            points = np.array(all_points, dtype='float64')

        if points is not None:
            self.points = np.array(points, dtype='float64')
        else:
            self.points = np.array([], dtype='float64')

        self.start = np.array(start, dtype='float64')

        self.color = format_color(color, alpha=fill_opacity)
        self.fill_opacity = fill_opacity
        self.fill = fill
        self.versions = []
        if not stroke:
            # under no conditions, stroke
            self.stroke = None
        else:
            # stroke only if stroke_width is defined
            self.stroke = False if not stroke_width else True

        self.stroke_width = stroke_width
        self.stroke_color = format_color(stroke_color,
                                         alpha=stroke_opacity) if stroke_color is not None else format_color(
            COLOR_DEFAULT, alpha=stroke_opacity)
        self.stroke_opacity = stroke_opacity
        self.line_cap = line_cap
        self.line_join = line_join
        self.fill_rule = fill_rule
        self.animate_at_center = True
        self.close_path = close_path
        self.start_end = []
        self.original = None
        self.as_bezier = True
        if anchor is not None:
            self.anchor = np.array(anchor, dtype='float64').copy()  # user defined for user-shape interaction
        else:
            self.anchor = self.start.copy()

        self.animate = Animate(self, ShapeAnimation)

    @property
    def kwargs(self):
        kwargs_dict = {
            'color': self.color,
            'fill_opacity': self.fill_opacity,
            'fill': self.fill,
            'stroke': self.stroke,
            'stroke_width': self.stroke_width,
            'stroke_color': self.stroke_color,
            'stroke_opacity': self.stroke_opacity,
            'line_cap': self.line_cap,
            'close_path': self.close_path
        }
        return kwargs_dict

    def set_kwargs(self, kwargs_dict):
        if 'fill_opacity' in kwargs_dict:
            self.fill_opacity = kwargs_dict['fill_opacity']
        if 'stroke_opacity' in kwargs_dict:
            self.stroke_opacity = kwargs_dict['stroke_opacity']
        if 'color' in kwargs_dict:
            self.color = format_color(kwargs_dict['color'], alpha=self.fill_opacity)
        if 'fill' in kwargs_dict:
            self.fill = kwargs_dict['fill']
        if 'stroke' in kwargs_dict:
            self.stroke = kwargs_dict['stroke']
        if 'stroke_width' in kwargs_dict:
            self.stroke_width = kwargs_dict['stroke_width']
        if 'stroke_color' in kwargs_dict:
            self.stroke_color = kwargs_dict['stroke_color']
        if 'close_path' in kwargs_dict:
            self.close_path = kwargs_dict['close_path']

    def copy(self):
        shape = Shape(points=self.points, start=self.start, color=self.color,
                     as_bezier=self.as_bezier, close_path=self.close_path, fill=self.fill, stroke=self.stroke,
                     stroke_width=self.stroke_width,  stroke_color=self.stroke_color,
                     stroke_opacity=self.stroke_opacity, anchor=self.anchor, fill_opacity=self.fill_opacity,
                     line_cap=self.line_cap, line_join=self.line_join)
        return shape

    def scale(self, x_s, y_s):
        if x_s != 1 or y_s != 1:
            start = self.anchor
            if x_s > 0:
                x_s_i = 1
            else:
                x_s_i = -1

            if y_s > 0:
                y_s_i = 1
            else:
                y_s_i = -1

            t_x, t_y = (start[0] * (x_s_i - x_s), start[1] * (y_s_i - y_s))  # determines how much to translate by
            self.points[:, 0] *= x_s
            self.points[:, 0] += t_x
            self.points[:, 1] *= y_s
            self.points[:, 1] += t_y
            self.start[0] *= x_s
            self.start[0] += t_x
            self.start[1] *= y_s
            self.start[1] += t_y

            if self.stroke_width:
                self.stroke_width *= max(x_s, y_s)

    def scale_to(self, p, s, given=False):
        """scales to position, size"""
        if given:
            # the scalars have been provided
            x_s, y_s = s
        else:
            w, h = s
            c_w, c_h = self.size
            x_s = w / c_w
            y_s = h / c_h
        self.scale(x_s, y_s)
        self.move_to(p)

    def move(self, m_d):
        m_x, m_y = m_d
        self.points[:, 0] += m_x
        self.points[:, 1] += m_y

        self.start += np.array([m_x, m_y])

    def move_to(self, c, center=True):
        n_x, n_y = c
        if center is True:
            start = self.center
        else:
            start = self.start

        a_x, a_y = start
        m_x, m_y = n_x - a_x, n_y - a_y
        self.points[:, 0] += m_x
        self.points[:, 1] += m_y

        self.start += np.array([m_x, m_y])

    def rotate(self, about_point, angle=0, clockwise=True):
        angle = format_angle(angle if clockwise else -angle)
        distance = self.points - about_point

        x_cos, y_sin, x_sin, y_cos = distance[:, 0] * math.cos(angle), distance[:, 1] * math.sin(angle), distance[:,
                                                                                                         0] * math.sin(
            angle), distance[:, 1] * math.cos(angle)

        self.points[:, 0] = x_cos - y_sin + about_point[0]
        self.points[:, 1] = x_sin + y_cos + about_point[1]
        start = self.start - about_point
        self.start = np.array([start[0] * math.cos(angle) - start[1] * math.sin(angle) + about_point[0],
                              start[0] * math.sin(angle) + start[1] * math.cos(angle) + about_point[1]])

    def reset(self):
        self.close_path = True
        self.as_bezier = True
        self.points = self.original.points.copy()

    def erase(self):
        if not self.original:
            self.original = self.copy()
        self.points = np.array([], dtype='float64')

    def save(self):
        self.original = self.copy()

    def create(self, alpha=0, do_close=None, step_size=FUNCTION_STEP):
        # shows a part of the creation of the shape up to alpha (between 0 and 1)
        # if not used, shape will draw fully
        self.erase()
        copied_alpha = alpha
        bezier_curves = self.original.path  # internal function to convert bezier points to equations

        self.as_bezier = False  # important for drawing as points and not bezier

        if alpha >= MARGIN_OF_ERROR and do_close in [None, True]:  # room for error
            print('here')
            self.reset()
            return
        else:
            self.close_path = False

        if alpha > 0:
            to_restore = []
            num_bezier = len(bezier_curves)
            curve_percent = 1 / num_bezier  # including starting point

            for curve in bezier_curves:
                if copied_alpha > curve_percent:
                    copied_alpha -= curve_percent
                    to_restore.extend(function_points(curve, 1, step_size=step_size))
                elif copied_alpha > 0:
                    to_restore.extend(function_points(curve, copied_alpha * num_bezier, step_size=step_size))
                    break
                else:
                    break
            self.points = np.array(to_restore, dtype='float64')

    def loop(self, n, pos_dif, row_wise=True, reverse=False):
        if isinstance(pos_dif, (list, tuple)):
            x_dif = pos_dif[0]
            y_dif = pos_dif[1]
        else:
            x_dif = y_dif = pos_dif

        if isinstance(n, int):
            if reverse:
                total_dif = -pos_dif
            else:
                total_dif = pos_dif
            loop_group = Group(self)
            for i in range(n - 1):
                looped_shape = self.copy()
                if row_wise:
                    looped_shape.move((total_dif, 0))
                else:
                    # column wise
                    looped_shape.move((0, total_dif))
                loop_group.add(looped_shape)
                if reverse:
                    total_dif -= pos_dif
                else:
                    total_dif += pos_dif
        elif isinstance(n, (list, tuple)):
            loop_group = Group()
            shape_copy = self.copy()
            if row_wise:
                x, y = n
            else:
                x, y = n[::-1]

            for i in range(x):
                loop_group.add(shape_copy.copy().loop(y, x_dif))
                shape_copy.move(0, y_dif)
        else:
            raise Exception('n type must be: int, list, or tuple')

        return loop_group

    def apply_matrix(self, func, at_point=None):
        if not at_point:
            at_point = np.array([0, 0])

        for i in range(len(self.points)):
            self.points[i] -= at_point
            self.points[i] = func(self.points[i])
            self.points[i] += at_point

        self.start -= at_point
        self.start = np.array(func(self.start))
        self.start += at_point

        self.anchor -= at_point
        self.anchor = func(self.anchor)
        self.anchor += at_point

    @property
    def flattened_points(self):
        return self.points  # this sort of property is necessary for more complex objects; such as the Path

    def get_extreme(self, idx_1, reverse=False):
        if hasattr(self, 'flattened_points') and len(self.flattened_points) > 0:
            if not isinstance(self.points, np.ndarray):
                return sorted([x[idx_1] for x in self.flattened_points], reverse=reverse)[0]
            return sorted(self.points[:, idx_1], reverse=reverse)[0]
        else:
            return 0

    @property
    def topmost(self):
        return self.get_extreme(1, False)

    @property
    def bottommost(self):
        return self.get_extreme(1, True)

    @property
    def rightmost(self):
        return self.get_extreme(0, True)

    @property
    def leftmost(self):
        return self.get_extreme(0, False)

    @property
    def size(self):
        """gets the size of the shape's approx. bounding box"""
        return np.array([self.rightmost-self.leftmost, self.bottommost-self.topmost], dtype=np.float64)

    @property
    def center(self):
        return np.array([(self.rightmost+self.leftmost)/2, (self.bottommost+self.topmost)/2], dtype=np.float64)

    @property
    def path(self):
        return shape_to_bezier_curves(self.start, self.points, close_path=self.close_path)


class CubicBezier(Shape):
    def __init__(self, *points, **kwargs):
        points = [flatten_list(x) for x in points]
        if 'fill' not in kwargs:
            kwargs['fill'] = False
        if 'close_path' not in kwargs:
            kwargs['close_path'] = False
        super().__init__(points[1:], points[0], **kwargs)


class Complex(Shape):
    """a complex shape, made from lists of points and/or other connected shapes"""
    def __init__(self, *complex, **kwargs):
        all_points = []
        if len(complex) == 1:
            '''if the Complex only has one element, assume it is a shape'''
            shape = complex[0]
            super().__init__(shape.points, shape.start, **kwargs)
        else:
            last_item = []  # last item, was shape; connects two shapes together
            start = None # the starting point
            for j, item in enumerate(complex):
                if isinstance(item, Shape):
                    connecting_line = []
                    item_points = item.points

                    if last_item:
                        connecting_line = line_to_cubic_bezier(*last_item, *item.start)
                    else:
                        start = item.start  # this is the first item, and it already has a start not defined in points
                    all_points.extend([*connecting_line, *item_points])
                    last_item = item_points[-1].tolist()

                elif isinstance(item, (list, tuple)):
                    if not nested(item):
                        item = [item]

                    if not start:
                        start = item[0]
                        if isinstance(start, np.ndarray):
                            start = start.tolist()

                    if len(item) == 1:
                        if last_item:
                            connecting_line = line_to_cubic_bezier(*last_item, *flatten_list(item))
                            all_points.extend([*connecting_line])
                        last_item = item[0]
                    elif len(item) == 2:
                        if last_item:  # <- expected behavior, , makes new line from previous point
                            first = item[0]
                            first_line = line_to_cubic_bezier(*last_item, *first)
                            all_points.extend(first_line)

                        connecting_line = line_to_cubic_bezier(*flatten_list(item))
                        all_points.extend(connecting_line)
                        last_item = item[-1]
                    elif len(item) == 3:
                        all_points.extend(item)
                        last_item = item[-1]
                    elif len(item) == 4:
                        all_points.extend(item[1:])
                        last_item = item[-1]

                    if isinstance(last_item, np.ndarray):
                        last_item = last_item.tolist()
            if isinstance(start, np.ndarray):
                start = start.tolist()
            start = all_points[0] if not start else start
            all_points = np.array(all_points, dtype='float64')
            super().__init__(all_points, start, **kwargs)


class Rectangle(Shape):
    """makes a rectangle"""
    def __init__(self, start_coordinate, size, color=COLOR_DEFAULT, **kwargs):

        a_x, a_y = start_coordinate
        width, height = format_size(size)
        b_x = a_x + width
        c_y = a_y + height
        start = [b_x, a_y]
        points = [
            [b_x, a_y],
            [a_x, a_y],
            [a_x, c_y],
            [b_x, c_y]
        ]
        kwargs['anchor'] = [a_x,
                            a_y]  # for user interaction, set the anchor coordinate (not used except for user functions) to top-left
        super().__init__(points, start, color, as_bezier=False, **kwargs)
        self.animate_at_center = False


class Square(Rectangle):
    """makes a square"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Arc(Shape):
    def __init__(self, center, size, color=COLOR_DEFAULT, start=0, change=90, clockwise=False, in_rad=False, is_diameter=True, **kwargs):
        """uses center parameterization (see SVGArc for endpoint parameterization)"""
        if is_diameter:
            diameter, _ = format_size(size)
            self.radius = diameter / 2
        else:
            self.radius, _ = format_size(size)
        if not in_rad:
            start = format_angle(start)
            change = format_angle(change)

        angle_pairs = split_angle_into_sections(start, change, PI / 2, clockwise=clockwise)
        points = []
        shape_start = None
        for i, x in enumerate(angle_pairs):
            start, path = self.get_arc_as_bezier_path(*x, center)
            if i == 0:
                shape_start = start
            points.extend(path)

        kwargs['anchor'] = center
        super().__init__(points, shape_start, color, **kwargs)

    def get_arc_as_bezier_path(self, start, angle, center):
        cos_start, sin_start = math.cos(start), math.sin(start)
        cos_end, sin_end = math.cos(angle), math.sin(angle)
        start = (cos_start * self.radius + center[0], center[1] - sin_start * self.radius)
        end = (cos_end * self.radius + center[0], center[1] - sin_end * self.radius)
        x1, y1 = start
        x4, y4 = end
        xc, yc = center
        ax = x1 - xc
        ay = y1 - yc
        bx = x4 - xc
        by = y4 - yc
        q1 = ax * ax + ay * ay
        q2 = q1 + ax * bx + ay * by
        denom = ax * by - ay * bx
        denom = denom if max(abs(denom), EPS) != EPS else EPS # makes sure the denominator is never too small
        k2 = (4 / 3) * (math.sqrt(2 * q1 * q2) - q2) / denom
        x2 = xc + ax - k2 * ay
        y2 = yc + ay + k2 * ax
        x3 = xc + bx + k2 * by
        y3 = yc + by - k2 * bx
        return [x1, y1], [[x2, y2], [x3, y3], [x4, y4]]


class Circle(Arc):
    def __init__(self, center, size, color=COLOR_DEFAULT, start=0, change=360, **kwargs):
        super().__init__(center, size, color, start, change, **kwargs)


class Ellipse(Arc):
    def __init__(self, center, size, color=COLOR_DEFAULT, start=0, change=360, **kwargs):
        if isinstance(size, (float, int)):
            size = [size, size]
        super().__init__(center, size, color, start, change, **kwargs)

        self.scale(1, size[1] / size[0])  # scale y direction -- matter of preference


def SVGArc(start_point, r, x_axis_rotation, large_arc_flag, sweep_flag, end_point, in_rad=False, **kwargs):
    # arc with svg parameters, endpoint parameterization (see https://www.w3.org/TR/SVG2/implnote.html)
    r_x, r_y = r
    if r_x < 0:
        r_x = -r_x
    if r_y < 0:
        r_y = -r_y

    if r_x == 0 or r_y == 0:  # treat as line segment in this case (use w3c docs)
        points = [start_point, start_point, end_point]
        return CubicBezier(points, **kwargs)
    else:  # treat as arc
        phi = format_angle(x_axis_rotation) if not in_rad else x_axis_rotation
        phi_cos = math.cos(phi)
        phi_sin = math.sin(phi)
        x_1, y_1 = start_point
        x_2, y_2 = end_point
        phi_matrix = np.array([
            [phi_cos, phi_sin],
            [-phi_sin, phi_cos]
        ])
        subtraction_matrix = np.array([
            [(x_1 - x_2)/2],
            [(y_1 - y_2)/2]
        ])
        midpoint_matrix = np.array([
            [(x_1 + x_2) / 2],
            [(y_1 + y_2) / 2]
        ])
        x_prime_1, y_prime_1 = phi_matrix.dot(subtraction_matrix)

        x_prime_1_2 = x_prime_1 ** 2
        y_prime_1_2 = y_prime_1 ** 2
        # rescale r_x and r_y accordingly (if necessary to get a fit)
        lmbda = x_prime_1_2 / (r_x ** 2) + y_prime_1_2 / (r_y ** 2)
        if lmbda > 1:
            sqrt_lmbda = math.sqrt(lmbda)
            r_x = sqrt_lmbda * r_x
            r_y = sqrt_lmbda * r_y
        r_x_2 = r_x ** 2
        r_y_2 = r_y ** 2
        c_sign = 1 if large_arc_flag != sweep_flag else -1
        result = (r_x_2 * r_y_2 - r_x_2 * y_prime_1_2 - r_y_2 * x_prime_1_2) / (r_x_2 * y_prime_1_2 + r_y_2 * x_prime_1_2)
        c_mul = c_sign * math.sqrt(
            result if abs(result) > 1 - MARGIN_OF_ERROR else abs(result)
        )

        c_prime = c_mul * np.array([(r_x * y_prime_1) / r_y,
                                    -(r_y * x_prime_1) / r_x])
        c_x_prime, c_y_prime = c_prime

        c = phi_matrix.T.dot(c_prime) + midpoint_matrix

        p_c_sub_1 = np.array([(x_prime_1 - c_x_prime) / r_x, (y_prime_1 - c_y_prime) / r_y]).squeeze()
        p_c_sub_2 = np.array([(-x_prime_1 - c_x_prime) / r_x, (-y_prime_1 - c_y_prime) / r_y]).squeeze()

        theta = angle_between_vectors(np.array([1, 0]),
                                      p_c_sub_1)
        delta = angle_between_vectors(p_c_sub_1, p_c_sub_2)

        if sweep_flag == 1 and delta > 0:
            delta -= TAU
        elif sweep_flag == 0 and delta < 0:
            delta += TAU

        er = Ellipse(c, (r_x, r_y), start=theta, change=delta, in_rad=True, is_diameter=False, **kwargs)
        if len(er.points):  # if the ellipse doesn't have points, then don't rotate it
            er.rotate(c, x_axis_rotation, clockwise=True)
        return er


class Star(Shape):
    # adapted from https://docs.manim.community/en/stable/_modules/manim/mobject/geometry/polygram.html#Star
    def __init__(self, center, outer, inner=None, n=5, density=2, start_angle=-PI/2,  **kwargs):
        self.start_angle = start_angle
        outer_radius = outer / 2
        inner_angle = PI / n
        if inner is None:
            if density <= 0 or density >= n / 2:
                raise ValueError(
                    f"Incompatible density {density} for number of points {n}",
                )

            outer_angle = TAU * density / n
            inverse_x = 1 - np.tan(inner_angle) * (
                    (np.cos(outer_angle) - 1) / np.sin(outer_angle)
            )

            inner_radius = outer_radius / (np.cos(inner_angle) * inverse_x)
        else:
            inner_radius = inner / 2

        outer_vertices = points_on_circle(
            n,
            center,
            outer_radius,
            start_angle=self.start_angle
        )
        inner_vertices = points_on_circle(
            n,
            center,
            inner_radius,
            start_angle=self.start_angle + inner_angle
        )

        points = []
        for pair in zip(outer_vertices, inner_vertices):
            points.extend(pair)

        start = points[0]
        kwargs['center'] = center

        super().__init__(points, start, as_bezier=False, **kwargs)


class Curve(Shape):
    def __init__(self, points, **kwargs):
        super().__init__(points=points[1:], start=points[0], **kwargs)