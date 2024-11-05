from twiddler.constants import *
from twiddler.helper import divide_chunks, split_bezier, transform_fix_shapes, function_points
import numpy as np
from twiddler.color.color import format_color
from twiddler.trackers import *


class ShapeAnimation:
    """animation class for shapes; objects are added to the canvas list of animations to play"""
    def __init__(self, animate, parent, kwargs):
        self.animate = animate  # the Animate organizer (this is the child of that; these are the animation objects, children from the parent Animate controller)
        self.parent = parent  # the parent shape or other thing we're animating
        self.animation_func = None
        self.kwargs = kwargs
        self.first_frame = 0  # sets the frame when the animation should start
        self.last_frame = 0  # when the animation should end
        self.animation_frames = 1  # number of frames in the animation
        self.animation_time = 0  # how long the animation should be in seconds
        self.animation_frames_left = 1  # the number of frames left in the animation
        self.animation_start_pos = None  # what frame of the animation to begin at
        self.alpha = 0  # determines movement of shapes (animation speed)
        self.set_animation_time(ANIMATION_TIME)  # sets the default animation time if not reset
        self.removed = False

    def set_animation_time(self, animation_time=None):
        self.animation_frames = FRAME_RATE * animation_time - 1
        if self.animation_frames < 1:
            self.animation_frames = 1
        self.animation_time = animation_time
        self.alpha = 1 / self.animation_frames

    def set_animation_time_from_num_frames(self, num_frames=None):
        self.animation_frames = num_frames - 1
        if self.animation_frames < 1:
            self.animation_frames = 1
        self.animation_time = self.animation_frames / FRAME_RATE
        self.alpha = 1 / self.animation_frames

    def create(self, alpha_total=0, close_path=None, step_size=FUNCTION_STEP, **kwargs):
        if self.animation_start_pos:
            alpha_total = self.alpha * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            alpha_total += self.alpha

        self.parent.create(alpha_total, do_close=close_path, step_size=step_size)
        self.kwargs = {'alpha_total': alpha_total, 'close_path': close_path, 'step_size': step_size}

    def move_along_path(self, paths, close_path, step_size, alpha_total=0, curve_percent=0, path_i=0, num_bezier=0, current_curve=None, **kwargs):
        # moves a shape according to a designated path of bezier equations
        # perhaps integrate with create code someday?

        if self.animation_start_pos:
            alpha_total = self.alpha * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            alpha_total += self.alpha

        if not curve_percent:
            if not close_path:
                paths = paths[:-1]
            num_bezier = len(paths)
            curve_percent = 1 / len(paths)

            current_curve = paths[0]

        if alpha_total > curve_percent + 1 - MARGIN_OF_ERROR:
            path_i += 1
            alpha_total -= curve_percent

            current_curve = paths[path_i]

        next_point = function_points(current_curve, alpha_total * num_bezier, step_size=step_size)[-1]

        self.parent.move_to(next_point)
        self.kwargs = {'paths': paths, 'close_path': close_path, 'step_size': step_size, 'alpha_total': alpha_total, 'curve_percent': curve_percent, 'path_i': path_i, 'num_bezier': num_bezier, 'current_curve': current_curve}

    def move_along_func(self, func, min_val, max_val, current=None, d_change=None, **kwargs):
        # moves a shape along a function
        if current is None:
            current = min_val

        if d_change is None:
            d_change = self.alpha * (max_val - min_val)

        if self.animation_start_pos:
            current += d_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            current += d_change

        point = func(current)
        self.parent.move_to(point)
        self.kwargs = {'func': func, 'min_val': min_val, 'max_val': max_val, 'current': current, 'd_change': d_change}

    def move(self, m_d, center=True, change=None, animation_target=None, **kwargs):
        # moves a shape to a target
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
            self.parent.start += change * self.animation_start_pos
            self.parent.anchor += change * self.animation_start_pos
            self.parent.points += change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.start += change
            self.parent.anchor += change
            self.parent.points += change
        self.kwargs = {'m_d': m_d, 'center': center, 'change': change, 'animation_target': animation_target}

    def move_to(self, point, center=True, change=None, animation_target=None, **kwargs):
        # moves a shape to a target
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
            self.parent.start += change * self.animation_start_pos
            self.parent.anchor += change * self.animation_start_pos
            self.parent.points += change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.start += change
            self.parent.anchor += change
            self.parent.points += change
        self.kwargs = {'point': point, 'center': center, 'change': change, 'animation_target': animation_target}

    def scale_to(self, new_start, new_size, original_pos, original_size, scale_change=None, scale=None, pos_change=None, pos_dif=None, **kwargs):
        """scales and moves a shape into position (at center)"""

        if scale_change is None:
            new_size = np.array(new_size, dtype='float64')
            resize = new_size / self.parent.size
            scale_change = (resize - 1) * self.alpha
            scale = 1

        if pos_change is None:
            new_start = np.array(new_start, dtype='float64')
            pos_change = (new_start - original_pos) * self.alpha
            pos_dif = 0

        if self.animation_start_pos:
            scale += scale_change * self.animation_start_pos
            pos_dif += pos_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            scale += scale_change
            pos_dif += pos_change

        self.parent.scale_to(original_pos + pos_dif, original_size * scale)
        self.kwargs = {'new_start': new_start, 'new_size': new_size, 'original_pos': original_pos, 'original_size': original_size, 'scale_change': scale_change, 'scale': scale, 'pos_change': pos_change, 'pos_dif': pos_dif}

    def transform(self, new_shape, start_change=None, point_change=None, color_change=None, same_size=False, **kwargs):
        # transforms one shape into another
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
        self.kwargs = {'new_shape': new_shape, 'start_change': start_change, 'point_change': point_change,
                       'color_change': color_change, 'same_size': same_size}

    def color(self, new_color, color_change=None, **kwargs):
        # changes the color of a shape
        if color_change is None:
            new_color = format_color(new_color)
            color_change = (new_color - self.parent.color) * self.alpha

        if self.animation_start_pos:
            self.parent.color += color_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            self.parent.color += color_change

        self.kwargs = {'new_color': new_color, 'color_change': color_change}

    def rotate(self, about_point, angle, repeat, clockwise, angle_change=None, **kwargs):
        # rotates a shape
        if angle_change is None:
            angle *= repeat
            angle_change = angle * self.alpha

        if self.animation_start_pos:
            self.parent.rotate(about_point, angle=angle_change * self.animation_start_pos, clockwise=clockwise)
            self.animation_start_pos = 0
        else:
            self.parent.rotate(about_point, angle=angle_change, clockwise=clockwise)

        self.kwargs = {'about_point': about_point, 'angle': angle, 'repeat': repeat,
                       'clockwise': clockwise, 'angle_change': angle_change}

    def move_point(self, idx, new_point, this_point=None, point_change=None, **kwargs):
        # moves a single point according to a tracker
        is_value_tracker = isinstance(new_point, ValueTracker)
        if this_point is None:
            if idx == 0:
                this_point = self.parent.start
            else:
                this_point = self.parent.points[idx]

        if is_value_tracker:
            point_change = new_point - this_point
        elif point_change is None:
            point_change = (new_point - this_point) * self.alpha

        if self.animation_start_pos:
            if not is_value_tracker:
                if idx == 0:
                    self.parent.start += point_change * self.animation_start_pos
                else:
                    self.parent.points[idx] += point_change * self.animation_start_pos
            else:
                if idx == 0:
                    self.parent.start += point_change
                else:
                    self.parent.points[idx] += point_change
            self.animation_start_pos = 0
        else:
            if idx == 0:
                self.parent.start += point_change
            else:
                self.parent.points[idx] += point_change

        self.kwargs = {'idx': idx, 'new_point': new_point, 'this_point': this_point, 'point_change': point_change}
