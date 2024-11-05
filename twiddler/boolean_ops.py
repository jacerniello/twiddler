from pathops import Path as SkiaPath
from pathops import PathVerb, intersection, difference, union, xor
from twiddler.shape import Shape
from twiddler.group import Group
from twiddler.helper import *
import numpy as np


def shape_to_skia_path(shape_group):
    path = SkiaPath()
    if isinstance(shape_group, Shape):
        draw_path(path, [shape_group.start], [shape_group.points])
    elif isinstance(shape_group, Group):
        start = []
        points = []
        for shape in shape_group.elements:
            start.append(shape.start)
            points.append(shape.points)
        draw_path(path, start, points)
    return path


def skia_path_to_shape(outpen):
    last_point = (0, 0)
    starting_point = []
    all_shapes = []
    all_points = []

    for path_verb, points in outpen:
        if path_verb == PathVerb.MOVE:
            starting_point = points[0]
            all_points.append(points[0])
        elif path_verb == PathVerb.CUBIC:
            all_points.extend(points)
        elif path_verb == PathVerb.LINE:
            new_point = points[-1]
            all_points.extend(line_to_cubic_bezier(*last_point, *new_point))
        elif path_verb == PathVerb.CLOSE:
            mid_point = get_midpoint(last_point, starting_point)
            all_points.extend([last_point, mid_point, starting_point])
            all_shapes.append(all_points)
            all_points = []
        if points:
            last_point = points[-1]

    if len(all_shapes) >= 1:
        return all_shapes


def draw_path(path, start, points):
    for i in range(len(points)):
        s, p = start[i], points[i].flatten().tolist()
        path.moveTo(s[0], s[1])
        while p:
            path.cubicTo(*p[:6])
            del p[:6]

    path.close()


class BooleanOperations(Group):
    def __init__(self, *args, **kwargs):
        self.all_kwargs = []
        self.current_i = 0
        self.as_group = True
        self.op = intersection  # default boolean operation
        self.is_union = False
        self.is_boolean_op = True
        super().__init__(**kwargs)

    def build_group(self, shapes):
        if not shapes:
            return
        shapes_group = Group()
        for shape_points in shapes:
            shape_points = np.array(shape_points, dtype='float64')
            if self.current_i > len(self.all_kwargs) - 1:  # in case of shape break, use style attributes for last shape
                self.current_i -= 1
            new_shape = Shape(shape_points[1:], shape_points[0], **self.all_kwargs[self.current_i])
            shapes_group.add(new_shape)
            self.current_i += 1
        return shapes_group

    def shape_pair_op(self, first, second):
        outpen = SkiaPath()

        self.op(
            [shape_to_skia_path(first)],
            [shape_to_skia_path(second)],
            outpen.getPen()
        )
        return outpen

    def multi_shapes(self, groups, **kwargs):
        all_groups = []
        all_outpens = []
        all_e_groups = [x.elements for x in groups]

        next_g = all_e_groups[1]
        for j, this_e in enumerate(all_e_groups[0]):
            for k, next_e in enumerate(next_g):
                out = self.shape_pair_op(this_e, next_e)
                if out:
                    self.all_kwargs.append(this_e.kwargs)

                all_outpens.append(out)

        for n_out in all_outpens:
            result = self.build_group(skia_path_to_shape(n_out))
            if result:
                all_groups.append(result)

        if len(groups) > 2:
            self.multi_shapes([Group(*all_groups), *groups[2:]], **kwargs)
        else:
            for item in all_groups:
                self.add(item)
        self.reset_kwargs()

    def reset_kwargs(self):
        self.all_kwargs = []
        self.current_i = 0


class Union(BooleanOperations):
    # already supports multi, so no need for multi_shapes
    def __init__(self, *shapes, **kwargs):
        super().__init__(**kwargs)
        self.shapes = shapes
        self.op = union
        self.is_union = True
        self.make_union()

    def make_union(self):
        all_elements = []
        for shape in self.shapes:
            if isinstance(shape, Group):
                all_elements.extend(shape.elements)
            else:
                all_elements.append(shape)

        for this_e in all_elements:
            self.all_kwargs.append(this_e.kwargs)
        outpen = SkiaPath()
        paths = [shape_to_skia_path(elem) for elem in all_elements]
        self.op(
            paths,
            outpen.getPen()
        )
        item = self.build_group(skia_path_to_shape(outpen))
        self.add(item)
        self.reset_kwargs()


def format_shapes(shapes):
    shapes = list(shapes)
    for i, shape in enumerate(shapes):
        if not isinstance(shape, Group):
            shapes[i] = Group(shape)
    return shapes


class Intersection(BooleanOperations):
    def __init__(self, *shapes, **kwargs):
        super().__init__(**kwargs)
        self.shapes = format_shapes(shapes)
        self.op = intersection
        self.multi_shapes(self.shapes, **kwargs)


def intersects(a, b):
    outpen = SkiaPath()

    a, b = [shape_to_skia_path(a)], [shape_to_skia_path(b)]
    intersection(
        a, b,
        outpen.getPen()
    )
    return len(outpen) > 0


class Difference(BooleanOperations):
    def __init__(self, *shapes, **kwargs):
        super().__init__(**kwargs)
        self.shapes = format_shapes(shapes)
        self.op = difference
        self.multi_shapes(self.shapes, **kwargs)


class Exclusion(BooleanOperations):
    def __init__(self, *shapes, **kwargs):
        super().__init__(**kwargs)
        self.shapes = format_shapes(shapes)
        self.op = xor
        self.multi_shapes(self.shapes, **kwargs)
