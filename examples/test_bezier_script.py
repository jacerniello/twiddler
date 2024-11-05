from twiddler import *


class Testing(Canvas):
    def construct(self):
        center_circle, c = SVGArc((100, 50), (50, 40), 80, 0,  0, (150, 100), fill=False, close_path=False, stroke_width=2, stroke_color='red')

        s = 5
        c_a = Circle((100, 50), s, color='white')
        c_b = Circle((150, 100), s, color='white')
        self.add(center_circle, c, c_a, c_b)


testing_file = '/Users/juliancerniello/PycharmProjects/video_maker/video/test.png'
config = Config(200, 200)
renderer = Testing(config=config)
renderer.render_frame(testing_file, time=1)
show_video(testing_file)