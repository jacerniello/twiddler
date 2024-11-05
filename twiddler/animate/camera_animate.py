import numpy as np
from twiddler.animate.shape_animate import ShapeAnimation


class CameraAnimation(ShapeAnimation):
    def __init__(self, animate, parent, kwargs):
        super().__init__(animate, parent, kwargs)

    def scale_to(self, new_start, new_size, translation_change=None, translation=None, scale_change=None, scale=None, resize=None, **kwargs):

        self.parent.restore()

        if scale_change is None:
            new_size = np.array(new_size, dtype='float64')
            new_start = np.array(new_start, dtype='float64')

            resize = self.parent.size / np.array(new_size, dtype='float64')

            scale_change = (resize - 1) * self.alpha

            scale = 1

            if hasattr(self.parent, 'stored_t'):
                final_start = new_start - self.parent.stored_t
            else:
                final_start = new_start
            translation_change = -np.array(final_start) * resize * self.alpha

            translation = 0

            # sets new size of the camera after the animation
            self.parent.size = new_size
            self.parent.stored_t = new_start * resize

        self.parent.save()
        if self.animation_start_pos:
            scale += scale_change * self.animation_start_pos
            translation += translation_change * self.animation_start_pos
            self.animation_start_pos = 0
        else:
            scale += scale_change
            translation += translation_change

        # print(scale, resize, scale[0], resize[0])

        animation_finished = (abs(resize - scale) < 0.0000001).all()

        if animation_finished:
            # print('y1')
            self.parent.restore()

        self.parent.canvas.translate(*translation)
        self.parent.canvas.scale(*scale)

        if animation_finished:
            # print('y2')
            # saves the camera's state at the end of the animation instead of resetting it
            self.parent.save()

        self.kwargs = {'new_start': new_start, 'new_size': new_size, 'translation_change': translation_change, 'translation': translation, 'scale_change': scale_change, 'scale': scale, 'resize': resize}
