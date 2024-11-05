import cairo
from twiddler.helper import *
from twiddler.color.color import format_color
from twiddler.animate.animate import GradientAnimate


def draw_gradient(gradient_obj, g_type='linear'):
    coordinates = flatten_list(gradient_obj.coordinates)
    if g_type == 'radial':
        c_gradient = cairo.RadialGradient(*coordinates)
    else:
        # type is linear or otherwise (such as another string or not defined)
        c_gradient = cairo.LinearGradient(*coordinates)

    if gradient_obj.repeat:
        c_gradient.set_extend(cairo.Extend.REPEAT)
    elif gradient_obj.reflect:
        c_gradient.set_extend(cairo.Extend.REFLECT)
    elif gradient_obj.pad:
        c_gradient.set_extend(cairo.Extend.PAD)

    for color_stop in gradient_obj.colors:
        color, stop = color_stop
        c_gradient.add_color_stop_rgba(stop, *color)

    return c_gradient


class LinearGradient:
    def __init__(self, *coordinates, step_size=0.1, repeat=False, reflect=False, pad=False):
        self.gradient = True
        self.coordinates = coordinates  # bounds
        self.colors = []
        self.step_size = step_size
        self.repeat = repeat
        self.reflect = reflect
        self.pad = pad
        self.animate = GradientAnimate(self)

    def add_colors(self, *colors):
        """add colors without step size"""
        formatted_colors = [[format_color(color), self.step_size * (i+1)] for i, color in enumerate(colors)]
        self.colors.extend(formatted_colors)

    def add_color_stops(self, *color_stops):
        for color_stop in color_stops:
            color, stop = color_stop
            color = format_color(color)
            self.colors.append([color, stop])

    def source(self):
        return draw_gradient(self, g_type='linear')


class RadialGradient:
    def __init__(self, *coordinates, step_size=0.1, repeat=False, reflect=False, pad=False):
        self.gradient = True
        self.coordinates = []
        for p in coordinates:
            pos, size = p
            if isinstance(size, (list, tuple)):
                self.coordinates.append([*pos, size[0]])
            else:
                self.coordinates.append([*pos, size])

        self.colors = []
        self.step_size = step_size
        self.repeat = repeat
        self.reflect = reflect
        self.pad = pad
        self.animate = GradientAnimate(self)

    def add_colors(self, *colors):
        formatted_colors = [[format_color(color), self.step_size * (i + 1)] for i, color in enumerate(colors)]
        self.colors.extend(formatted_colors)

    def add_color_stops(self, *color_stops):
        self.step_size = False
        for color_stop in color_stops:
            color, stop = color_stop
            color = format_color(color)
            self.colors.append([color, stop])

    def source(self):
        return draw_gradient(self, g_type='radial')

