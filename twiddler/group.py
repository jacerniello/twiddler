import numpy as np
from twiddler.animate.animate import Animate, GroupAnimation


class Group:
    def __init__(self, *shapes, animate=True, auto_center=True, **kwargs):
        # contains groups of shapes and can be used to move more than one shape
        # also controls a shape's

        self.elements = []
        self.start_end = []
        if auto_center:
            self.anchor = self.center
        else:
            self.anchor = self.elements[0].anchor if self.elements else np.array([0, 0])

        self.kwargs = kwargs
        self.animate_group = animate
        if self.animate_group:
            self.animate = Animate(self, animation_center=self.anchor, animation_type=GroupAnimation)

        self.add(*shapes)
        self.distribute_kwargs(shapes)

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.flattened_elements)

    def distribute_kwargs(self, shapes):
        for shape in shapes:
            shape.set_kwargs(self.kwargs)

    def add(self, *shapes):
        for shape in shapes:
            self.elements.append(shape)

        self.distribute_kwargs(self.elements)
        if self.animate_group:
            self.animate.animation_center = self.center

    def remove(self, i):
        if isinstance(i, int):
            if i == -1:
                self.elements = []
            else:
                del self.elements[i]
        else:
            if i in self.elements:
                self.elements.remove(i)

        if self.animate_group:
            self.animate.animation_center = self.center

    @property
    def center(self):
        r, l, t, b = self.rightmost, self.leftmost, self.topmost, self.bottommost
        if None not in [r, l, t, b]:
            return np.array([(r + l) / 2, (b + t) / 2], dtype=np.float64)

    @property
    def bottommost(self):
        all_elems = [x.bottommost for x in self.flattened_elements]
        if not all_elems:
            return
        return max(all_elems)

    @property
    def topmost(self):
        all_elems = [x.topmost for x in self.flattened_elements]
        if not all_elems:
            return
        return min(all_elems)

    @property
    def leftmost(self):
        all_elems = [x.leftmost for x in self.flattened_elements]
        if not all_elems:
            return
        return min(all_elems)

    @property
    def rightmost(self):
        all_elems = [x.rightmost for x in self.flattened_elements]
        if not all_elems:
            return
        return max(all_elems)

    @property
    def size(self):
        """gets the size of the group's approx. bounding box"""
        return np.array([self.rightmost - self.leftmost, self.bottommost - self.topmost], dtype=np.float64)

    @property
    def color(self):
        """retrieves the color of a group element"""
        return np.array([x.color for x in self.flattened_elements], dtype=np.float64)

    @color.setter
    def color(self, value):
        """sets the color of a group"""
        if value.ndim == 1:
            for idx, x in enumerate(self.flattened_elements):
                x.color = value
        else:
            for idx, x in enumerate(self.flattened_elements):
                x.color = value[idx]

    def move(self, m_d, **kwargs):
        for elem in self.flattened_elements:
            elem.move(m_d, **kwargs)

    def move_to(self, pos, old_pos=None, center=True):
        if not self.flattened_elements:
            return

        if (isinstance(center, np.ndarray) and center.all()) or center:
            change = np.array(pos) - self.center
        else:
            change = np.array(pos) - np.array(old_pos)

        for x in self.flattened_elements:
            if isinstance(x, Group):
                x.move_to(pos, old_pos=old_pos, center=center)
            else:
                for i in range(len(x.points)):
                    x.points[i] += change

                x.start += change
                x.anchor += change

    def scale(self, x_s, y_s, **kwargs):
        for element in self.flattened_elements:
            prev_center = element.center
            element.scale(x_s, y_s, **kwargs)
            element.move_to(prev_center * [x_s, y_s])

    def rotate(self, about_point, angle):
        """group rotation"""
        for e in self.flattened_elements:
            e.rotate(about_point, angle)

    def scale_to(self, p, s, given=False, **kwargs):
        if given:
            # the scalars have been provided
            x_s, y_s = s
        else:
            w, h = s
            c_w, c_h = self.size
            x_s = w / c_w
            y_s = h / c_h
        self.scale(x_s, y_s, **kwargs)
        self.move_to(p)

    def set_kwargs(self, kwargs):
        for elem in self.flattened_elements:
            elem.set_kwargs(kwargs)

    def erase(self):
        for elem in self.flattened_elements:
            elem.erase()

    def apply_matrix(self, *args, **kwargs):
        for elem in self.flattened_elements:
            elem.apply_matrix(*args, **kwargs)

    @property
    def flattened_elements(self):
        """gets the base elements of the group"""
        elements = []
        for x in self.elements:
            if isinstance(x, Group):
                elements.extend(x.flattened_elements)
            else:
                elements.append(x)
        return elements

    def copy(self, g=None):
        """creates a completely identical group, with new elements. uses recursion"""
        layered_g = []

        to_check = self.elements if not g else g
        for e in to_check:
            if isinstance(e, Group):
                layered_g.append(self.copy(e))
            else:
                layered_g.append(e.copy())

        return Group(*layered_g)
