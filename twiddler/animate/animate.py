from twiddler.animate.shape_animate import ShapeAnimation
from twiddler.animate.group_animate import GroupAnimation
from twiddler.animate.gradient_animate import GradientAnimation
from twiddler.animate.camera_animate import CameraAnimation
from twiddler.helper import format_angle
from twiddler.trackers import ValueTracker
from twiddler.constants import *
import numpy as np

"""Animate objects are managers for Animation objects; each shape, gradient, or camera is attached to a single Animate object that creates linked Animations (with shared attributes)"""


class AnimateConfig:
    # manages all the Animation objects (creates them, acts as their parent and stores/shares important variables each Animation object will need to access or have reference to)
    def __init__(self, parent, animation_type):
        self.wait_counter = 0  # wait counter forces a wait period between animations
        self.offset = 0  # offset ensures that the animations for a current object are run sequentially
        self.num_running = 0  # determines the number of running animations for a given object; will return an error in canvas if there is more than 1
        self.parent = parent  # the shape or other object we're animating
        self.animation_type = animation_type  # the type of animations this Animate object can create


class Animate(AnimateConfig):
    def __init__(self, parent, animation_type, animation_center=None):
        super().__init__(parent, animation_type)

        if animation_center is not None:
            self.animation_center = animation_center
        else:
            self.animation_center = self.parent.anchor if hasattr(self.parent,
                                                                  'anchor') else self.parent.center if hasattr(
                self.parent, 'center') else None

    def move(self, m_d):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'m_d': m_d}
                                            )
        new_animation.animation_func = new_animation.move
        return new_animation

    def move_to(self, val):
        if hasattr(val, 'anchor'):
            point = val.anchor
        elif isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
            point = np.array(val, dtype="float64")
        elif isinstance(val, ValueTracker):
            point = val
        else:
            raise Exception('Need valid input for move_to (either a shape or point)')

        new_animation = self.animation_type(self,
                                            self.parent,
                                            {"point": point})
        new_animation.animation_func = new_animation.move_to
        return new_animation

    def scale_to(self, c, s):
        original_size = self.parent.size
        new_animation = ShapeAnimation(self,
                                       self.parent,
                                       {'new_start': c, 'new_size': s, 'original_pos': self.parent.center,
                                        'original_size': original_size}
                                       )  # always ShapeAnimation,  this animation can handle both groups and shapes
        new_animation.animation_func = new_animation.scale_to
        return new_animation

    def transform(self, new_shape):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'new_shape': new_shape}
                                            )
        new_animation.animation_func = new_animation.transform
        return new_animation

    def color(self, new_color):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'new_color': new_color}
                                            )
        new_animation.animation_func = new_animation.color
        return new_animation

    def move_point(self, idx, new_point):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'idx': idx, 'new_point': new_point})
        new_animation.animation_func = new_animation.move_point
        return new_animation

    def rotate(self, about_point, start_angle=0, angle=360, repeat=1, clockwise=True):
        if start_angle != 0:
            self.parent.rotate(about_point, angle=start_angle,
                               clockwise=False)  # start angle is always from right to left, like how an arc is drawn

        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'about_point': about_point,
                                             'angle': angle,
                                             'clockwise': clockwise,
                                             'repeat': repeat}
                                            )
        new_animation.animation_func = new_animation.rotate
        return new_animation

    def create(self, close_path=True, step_size=FUNCTION_STEP):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'close_path': close_path, 'step_size': step_size}
                                            )
        new_animation.animation_func = new_animation.create
        self.parent.erase()
        return new_animation

    def move_along_path(self, paths, close_path=False, step_size=FUNCTION_STEP):
        # close_path determines whether the animated shape returns to the beginning of the path once finished
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'paths': paths, 'close_path': close_path, 'step_size': step_size}
                                            )
        new_animation.animation_func = new_animation.move_along_path
        return new_animation

    def move_along_func(self, func, min_val, max_val):
        # moves along a function (function gives points)
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'func': func, 'min_val': min_val, 'max_val': max_val}
                                            )
        new_animation.animation_func = new_animation.move_along_func
        return new_animation

    def bool_op(self):
        if self.animation_type == GroupAnimation:
            new_animation = self.animation_type(self,
                                                self.parent,
                                                {}
                                                )
            if hasattr(self.parent, 'is_boolean_op') and self.parent.is_boolean_op:
                if self.parent.is_union:
                    new_animation.animation_func = new_animation.union
                else:
                    new_animation.animation_func = new_animation.bool_op
                return new_animation
            else:
                raise Exception(
                    'bool_op animation only works on boolean operation (intersection, difference, ...) objects')

        else:
            raise Exception(
                "Shape objects are not supported with the bool_op animation... must be a boolean operation object")


class GradientAnimate(AnimateConfig):
    def __init__(self, parent):
        super().__init__(parent, GradientAnimation)

    def move_gradient_to(self, *target_positions):
        """moves a gradient to a new position"""
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'target_positions': target_positions}
                                            )
        new_animation.animation_func = new_animation.move_gradient_to
        return new_animation

    def transform_gradient(self, g_new):
        """transforms one gradient into another (assuming same type, linear or radial)"""
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'g_new': g_new}
                                            )
        new_animation.animation_func = new_animation.transform_gradient
        return new_animation


class CameraAnimate(AnimateConfig):
    def __init__(self, parent):
        super().__init__(parent, CameraAnimation)

    def scale_to(self, new_start, new_size):
        new_animation = self.animation_type(self,
                                            self.parent,
                                            {'new_start': new_start, 'new_size': new_size}
                                            )
        new_animation.animation_func = new_animation.scale_to
        return new_animation
