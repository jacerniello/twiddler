import cairo
import webcolors
import colorsys
import numpy as np
import random
from twiddler.helper import *


# format color
def format_color(color, alpha=1):  # color is a tuple
    interpret_func = lambda func_name, string: string.replace(' ', '').split(func_name)[1].split('(')[1].split(')')[
        0].split(',')

    already_normalized = False
    if isinstance(color, str):
        color = color.strip().lower()
        if len(color) == 0:
            color = COLOR_DEFAULT
        if color[0] == '#':
            # assume hex
            main_hex = color[:7]
            rgb = list(webcolors.hex_to_rgb(main_hex))
        elif 'hsl' in color:
            # assume hsl
            divide_by = [360, 100, 100]
            color = interpret_func('hsl', color)
            color[1], color[2] = color[2], color[1]
            components = [int(x) / divide_by[i] for i, x in enumerate(color)]
            rgb = list(colorsys.hls_to_rgb(*components))
            already_normalized = True
        elif 'rgb' in color:
            if '%' in color:
                rgb = [float(2.55 * float(x.replace('%', ''))) for x in interpret_func('rgb', color)]
            else:
                rgb = [float(x) for x in interpret_func('rgb', color)]
        else:
            # assume color name
            if 'gradient' in color:
                # gradients not yet supported ...
                rgb = [0, 0, 0]
            else:
                rgb = list(webcolors.name_to_rgb(color))
    elif isinstance(color, (np.ndarray, list, tuple)):
        rgb = color[:3]  # gets first three values in list
        if len(color) == 4:
            # if alpha is defined as something but the default is set, redefine alpha here to what that new value is
            if alpha == 1:
                alpha = color[3]
            else:
                # otherwise, set the new alpha to the (not default) alpha given in the function; the function alpha overrides
                color[3] = alpha

        if all([x <= 1.00001 for x in color]):  # .00001 allows for some error (for example, color animations)
            if len(color) == 3:
                # in this case, assume that something has gone wrong, so just return all zeroes
                return np.zeros(4, dtype='float64')
            return np.array(color, dtype='float64')
    elif hasattr(color, 'gradient') and color.gradient:
        # if this is a color gradient object, return that
        return color
    else:
        rgb = [0, 0, 0]  # sets default color to black
    if already_normalized:
        normalized_rgb = rgb
    else:
        normalized_rgb = [x / 255 for x in rgb]
    normalized_rgb.append(alpha)
    
    return np.array(normalized_rgb, dtype='float64')


def random_color(lower=None, upper=None, opacity=1):
    if isinstance(lower, np.ndarray):
        lower = lower.tolist()
    if not lower:
        color = [random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255), opacity]
        return format_color(color)
    elif not upper:
        lower_bound = format_color(lower)
        opacity = lower_bound[3]
        color = [random.uniform(lower_bound[0], 255), random.uniform(lower_bound[1], 255), random.uniform(lower_bound[2], 255), opacity]
        return format_color(color)
    elif upper and not lower:
        upper_bound = format_color(upper)
        opacity = upper[3]
        color = [random.uniform(0, upper_bound[0]), random.uniform(0, upper_bound[1]),
                 random.uniform(0, upper_bound[2]), opacity]
        return format_color(color)
    else:
        lower_bound = format_color(lower)
        upper_bound = format_color(upper)
        opacity = upper_bound[3]
        color = [random.uniform(lower_bound[0], upper_bound[0]), random.uniform(lower_bound[1], upper_bound[1]),
                 random.uniform(lower_bound[2], upper_bound[2]), opacity]
        return format_color(color)


def random_brightness(color, bounds):
    color = format_color(color)
    brightness_change = random.uniform(*bounds)
    color[:3] = [x * brightness_change for x in color[:3]]
    return color


def make_g_obj(gradient_type, points, colors):
    if gradient_type == 'l_gradient':
        g = cairo.LinearGradient(*points)
    elif gradient_type == 'r_gradient':
        g = cairo.RadialGradient(*points)
    else:
        raise Exception('Not a valid gradient type!')
    for color in colors:
        s, c_f, a = color
        if len(c_f) == 4:  # if it is rgba
            g.add_color_stop_rgba(s, *c_f)
        else:
            g.add_color_stop_rgba(s, *c_f, a)
    return g
