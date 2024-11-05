import math
from twiddler.constants import *
import itertools
import numpy as np
import os


def angle_between_vectors(u, v, use_sign=True):
    """angle between 2d vectors"""
    u_x, u_y = u
    v_x, v_y = v
    if use_sign:
        sign = -1 if u_x * v_y - u_y * v_x > 0 else 1
    else:
        sign = 1
    dot_product = np.dot(u, v)
    return sign * np.arccos(dot_product / (np.linalg.norm(u) * np.linalg.norm(v)))


def split_angle_into_sections(init_a, change, q, clockwise=False):
    """splits an angle into sections"""
    a = float(init_a)
    b = float(change)
    sections = []

    n = 1
    current_angle = init_a

    if clockwise:
        # go from big -> small
        if change > 0:
            # 0 to 90 (clockwise)
            while change >= q:
                sections.append([current_angle, current_angle - q])
                change -= q
                current_angle -= q
                n += 1
            if change > 0:
                sections.append([current_angle, current_angle - change])
        else:
            # 0 to -90 (clockwise)
            q = -q
            while change <= q:
                sections.append([current_angle, current_angle - q])
                change -= q
                current_angle -= q
                n += 1
            if change < 0:
                sections.append([current_angle, current_angle - change])

    else:
        # go from small -> big
        # 0 to 90
        # 0 to -90
        if change > 0:
            # 0 to 90 (positive counterclockwise)
            while change >= q:
                sections.append([current_angle, current_angle + q])
                change -= q
                current_angle += q
                n += 1
            if change > 0:
                sections.append([current_angle, current_angle + change])
        else:
            # 0 to -90 (negative counterclockwise)
            q = -q

            while change <= q:
                sections.append([current_angle, current_angle + q])
                change -= q
                current_angle += q
                n += 1
            if change < 0:
                sections.append([current_angle, current_angle + change])

    return sections


def format_size(size):
    if isinstance(size, (list, tuple, np.ndarray)) and len(size) == 2:
        return size
    elif isinstance(size, (int, float)):
        size = [size] * 2
        return size
    else:
        raise Exception('Invalid type for argument size')


def format_angle(angle):
    if INPUT_IN_DEGREES:
        return angle * math.pi / 180
    else:
        return angle


def reformat_angles(angle_list):
    reformatted = []
    for angle in angle_list:
        reformatted.append(format_angle(angle))
    return reformatted

def get_midpoint(a, b):
    return (float(a[0]) + float(b[0])) / 2, (float(a[1]) + float(b[1])) / 2


def line_to_cubic_bezier(x1, y1, x2, y2):
    # converts a line of two points to the cubic bézier curve format by averaging the two points and setting that as a control point
    h_x, h_y = get_midpoint((x1, y1), (x2, y2))
    return np.array([[x1, y1], [h_x, h_y], [x2, y2]], dtype='float64')


def nested(l):
    return any(isinstance(i, (list, tuple, np.ndarray)) for i in l)


def flatten_list(l):
    if nested(l):
        return [item for sublist in l for item in sublist]
    else:
        return l


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def all_pairs(lst):
    c = list(itertools.combinations(lst, 2))

    return c


def pairwise_split(l):
    all_pairs = []
    for v, w in zip(l[:-1], l[1:]):
        all_pairs.append([v, w])
    return all_pairs


def split_bezier(points, t=0.5):
    # P_000, P_001, P_011, P_111 = [np.array(x, dtype='float64') for x in points]

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    x12 = (x2 - x1) * t + x1
    y12 = (y2 - y1) * t + y1

    x23 = (x3 - x2) * t + x2
    y23 = (y3 - y2) * t + y2

    x34 = (x4 - x3) * t + x3
    y34 = (y4 - y3) * t + y3

    x123 = (x23 - x12) * t + x12
    y123 = (y23 - y12) * t + y12

    x234 = (x34 - x23) * t + x23
    y234 = (y34 - y23) * t + y23

    x1234 = (x234 - x123) * t + x123
    y1234 = (y234 - y123) * t + y123

    return [(x1, y1), (x12, y12), (x123, y123), (x1234, y1234), (x234, y234), (x34, y34), (x4, y4)]


def points_to_bezier(points):
    all_curves = []
    start, points = points[0], points[1:]
    last = None
    for chunk in divide_chunks(points, 3):
        if last is not None:
            curve = [*chunk, last]
        else:
            curve = [start, *chunk]
        last = chunk[-1]
        all_curves.append(curve)

    return all_curves


def transform_fix(item, target):
    a_curves = divide_chunks(item.points, 3)
    all_curves = []
    last_point = item.start
    remaining = len(item.points)
    for i, curve in enumerate(a_curves):
        if len(all_curves) + remaining < target:
            curve = curve.tolist()
            curve.insert(0, last_point)

            v = split_bezier(curve)

            all_curves.extend(v[1:])

            last_point = v[-1]
        elif len(all_curves) < target and remaining > 0:
            all_curves.extend(curve)
        else:
            break
        remaining -= 1

    result = np.array(all_curves, dtype='float64')
    return result


def transform_fix_shapes(a, b):
    a_shape = a.points.shape[0]
    b_shape = b.points.shape[0]
    if a_shape > b_shape:
        target = a_shape
        result = np.array([])
        while result.shape[0] < target:
            result = transform_fix(b, target)
            b.points = result
    elif b_shape > a_shape:
        target = b_shape
        result = np.array([])
        while result.shape[0] < target:
            result = transform_fix(a, target)
            a.points = result
    else:
        return


def points_on_circle(n, center, radius, start_angle=0):
    iter_angle = TAU / n
    x_c, y_c = center
    points = [
        [radius * math.cos(iter_angle * x + start_angle) + x_c, radius * math.sin(iter_angle * x + start_angle) + y_c]
        for x in range(n)
    ]
    return points


def cubic_bezier_curve(points):
    return (
        lambda t: (1 - t) ** 3 * points[0]
                  + 3 * t * (1 - t) ** 2 * points[1]
                  + 3 * (1 - t) * t ** 2 * points[2]
                  + t ** 3 * points[3]
    )


def shape_to_bezier_curves(start, points, as_equations=True, close_path=True):
    new_points = list(points)
    start_chunks = list(divide_chunks(new_points, 3))
    all_bezier = []
    a = 0
    for b in start_chunks:
        if a == 0:
            b.insert(0, start)
            a = 2
        else:
            b.insert(0, new_points[a])
            a += 3
        all_bezier.append(b)

    if close_path:
        all_bezier.append(
            [all_bezier[-1][-1], *line_to_cubic_bezier(*all_bezier[-1][-1], *start)]
        )

    if as_equations:
        all_bezier = [cubic_bezier_curve(x) for x in all_bezier]

    return all_bezier


def function_points(equation, max_val, step_size=FUNCTION_STEP):
    # converts an equation to a list of points
    # max_val: max value for the function to take
    points = []
    x = 0
    while x < max_val:
        result = equation(x)
        points.append(result)
        x += step_size
    points = np.array(points, dtype='float64')
    return points


def make_new_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)


def array_round(a, r):
    """round an array"""
    if isinstance(a, np.float64):
        a = round(a, 8)
    else:
        for i in range(len(a)):
            a[i] = round(a[i], r)
    return a


def dist(a, b):
    dist = np.linalg.norm(a - b)
    return dist