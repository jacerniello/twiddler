from twiddler import *


class Scene(Canvas):
    def construct(self):
        circle = Circle(self.canvas_center, (500, 500), color="red")
        rectangle = Rectangle((100,100), (300, 200), color='dodgerblue')
        self.add(circle, rectangle)
        self.play(
            rectangle.animate.transform(circle)
        )


filename = "simple_shape.mp4"
scene_obj = Scene()
scene_obj.render(filename, time=1)
show_video(filename)
