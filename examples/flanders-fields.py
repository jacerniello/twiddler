from twiddler import *
from archive.custom_assets.kurzgesagt.clouds import generate_cloud
from archive.custom_assets.shapes import cross
import random
from PIL import Image
from functools import partial

from twiddler.shape import Shape, Rectangle
from twiddler.boolean_ops import Intersection
import numpy as np
import math


def HorizontalLoop(shape, start_bound, end_bound, height, repeat, **kwargs):

    bounds = Rectangle((start_bound, 0), (end_bound - start_bound, height), color='pink')
    looped_shape = shape.loop(math.ceil(repeat) + 1, end_bound - start_bound, reverse=True)
    intersection = Intersection(
        looped_shape, bounds
    )

    dist = (end_bound - start_bound) * repeat
    # print(dist)

    animations = [intersection.animate.bool_op(), looped_shape.animate.move_to(
        looped_shape.center + np.array([dist, 0], dtype='float64'))]

    return intersection, animations


def loop_x(val, canvas_width):
    if val[0] >= canvas_width:
        val[0] -= canvas_width
    return val



def poppy(flower_center, stem_width, stem_stroke_width, offset_a, offset_b):
    stem_stroke_color = random_color('#93c47d', 'darkgreen')
    a_a, a_b = offset_a
    b_a, b_b = offset_b
    stem = CubicBezier(flower_center,
                       (flower_center[0] + a_a, flower_center[1] + a_b),
                       (flower_center[0] + b_a, flower_center[1] + b_b),
                       (flower_center[0], flower_center[1] + stem_width), stroke_width=stem_stroke_width,
                       stroke_color=stem_stroke_color)
    lower = random_color('#fc6868', '#ff2626')
    flower_g = Group(
        Circle(flower_center, 100, color=lower),
        Circle(flower_center, 80, color=random_color(lower, '#ff2626')),
        Circle(flower_center, 20, color=random_color('#2d2d2d', 'black'))
    )

    return stem, flower_g


def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array


def generate_circles(n_w, n_h, image_array, block_size, radius):
    """ Approximate an image with circles of random size and color, allowing for overlap """
    out_circles = []
    h, w, _ = image_array.shape
    w_s = n_w / w
    h_s = n_h / h

    if w_s < h_s:
        h_s = w_s
        radius *= w_s
    if h_s < w_s:
        w_s = h_s
        radius *= h_s
    center_off_x = (n_w - w_s * w) / 2
    center_off_y = (n_h - h_s * h) / 2
    num_circles = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image_array[i:i + block_size, j:j + block_size]
            avg_color = tuple(np.mean(block, axis=(0, 1)).astype(int))

            center_x, center_y = j, i
            out_circles.append(
                Circle((center_x * w_s + center_off_x, center_y * h_s + center_off_y), radius, avg_color))
            num_circles += 1
    # print("# number of circles", num_circles)
    return out_circles, radius

def dist(a, b):
    a_0, a_1 = a
    b_0, b_1 = b
    return ((b_0 - a_0) ** 2 + (b_1 - a_1) ** 2) ** (1/2)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class Scene(Canvas):
    def poppies(self):
        num_poppies = 100
        all_poppies = Group()
        all_stems = Group()
        all_v = []
        for _ in range(num_poppies):
            break_all = True
            while break_all:
                rand_x = random.randint(100, 3740)
                rand_y = random.randint(1300, 2060)
                if all_v:
                    for i, x in enumerate(all_v):

                        if dist(x, np.array([rand_x, rand_y])) < 100:
                            break
                        if i == len(all_v) - 1:
                            break_all = False
                            break
                else:
                    break
            all_v.append([rand_x, rand_y])
            new_stem, new_poppy = poppy((rand_x, rand_y),
                                        stem_width=random.randint(120, 140),
                                        stem_stroke_width=10,
                                        offset_a=(random.randint(-20, 20), 60),
                                        offset_b=(0, 100)
                                        )

            all_stems.add(new_stem)
            all_poppies.add(new_poppy)

        animation_time = 200
        i_time = 2
        for i, poppy_group in enumerate(all_poppies.elements):
            stem = all_stems.elements[i]
            self.play(
                stem.animate.move_point(0, ValueTracker(poppy_group, 'center')),
                time=animation_time
            )
            change = np.array([random.randint(4, 10), random.randint(-10, 10)])
            for j in range(int(animation_time / i_time)):
                self.play(
                    poppy_group.animate.rotate(poppy_group.center + change, angle=360, clockwise=True),
                    time=i_time
                )

        return Group(all_stems, all_poppies)

    def sun(self):
        offset = 200
        self.sun_bg = Rectangle((-offset, -offset), (self.canvas_width+offset, self.canvas_height+offset))
        sun_center = np.array([1920, 1500])
        sun_color = RadialGradient((sun_center, 100),
                                   (sun_center, 2400))  # FDFBD3
        sun_color.add_color_stops(
            ['#ffe603', 0.1],
            ['#b8f3ff', 0.2],
            ['#a5e5ff', 0.6]
        )
        self.sun_bg.color = sun_color

        self.play(
            sun_color.animate.move_gradient_to(
                [1920, 500, 100],
                [1920, 500, 2400]
            ), time=4
        )
        return self.sun_bg

    def field(self):
        field_1 = Complex(
            [(0, 1600), (1000, 1700), (1300, 1500), (3840, 1600)],
            [(3840, 2160), (0, 3840)], color='#93c47d'
        )
        field_2 = Complex(
            [(0, 1200), (800, 1100), (1400, 1300), (3840, 1200)],
            [(3840, 2160), (0, 2160)], color='#274e13'
        )
        field_3 = Complex(
            [(0, 1100), (1000, 800), (1200, 1200), (3840, 1100)],
            [(3840, 2160), (0, 2160)],
            color='#674ea7'
        )
        all_fields = Group(field_3, field_2, field_1)
        return all_fields

    def crosses(self):
        """makes the two rows of crosses"""
        bg = "white"
        r1_y = 1600
        self.first_row = Group(
            cross(center=(600, r1_y),
                  thickness=300,
                  center_offset_vertical=100,
                  cross_width=800,
                  cross_height=1200, color=bg),
            cross(center=(2100, r1_y - 40),
                  thickness=300,
                  center_offset_vertical=100,
                  cross_width=800,
                  cross_height=1200, color=bg),
            cross(center=(3600, r1_y + 40),
                  thickness=300,
                  center_offset_vertical=100,
                  cross_width=800,
                  cross_height=1200, color=bg)
        )

        r2_y = 1100
        self.second_row = Group(
            cross(center=(0, r2_y),
                  thickness=120,
                  center_offset_vertical=60,
                  cross_width=350,
                  cross_height=500, color=bg),
            cross(center=(1300, r2_y),
                  thickness=120,
                  center_offset_vertical=60,
                  cross_width=350,
                  cross_height=500, color=bg),

            cross(center=(2900, r2_y + 20),
                  thickness=120,
                  center_offset_vertical=60,
                  cross_width=350,
                  cross_height=500, color=bg)
        )

        return self.first_row, self.second_row

    def bg(self):
        self.filter = Rectangle((0, 0), (self.canvas_width, self.canvas_height), color=(0, 0, 0, 0.9))
        self.play(self.filter.animate.color((0, 0, 0, 0)), time=4)
        return self.filter

    def bg_2(self):
        offset = 200
        self.filter = Rectangle((-offset, -offset), (self.canvas_width+offset, self.canvas_height+offset), color=format_color("brown", alpha=0))
        # self.play(self.filter.animate.color(format_color('brown', 0.3)))
        return self.filter

    def clouds(self, num_clouds):
        c_x = 0
        x_c = self.canvas_width / num_clouds
        all_r = []
        for _ in range(num_clouds):
            random_point = (random.uniform(0, self.canvas_width), random.uniform(-300, 600))
            a = generate_cloud(random_point)
            a.color = random_brightness('white', bounds=[0.85, 1])
            a.color[3] = random.uniform(0.7, 0.9)  # <-- opacity
            r, r_animations = HorizontalLoop(a, start_bound=0, end_bound=self.canvas_width, height=self.canvas_height,
                                             repeat=random.uniform(2, 4))
            all_r.append(r)
            self.play(
                *r_animations,
                time=40
            )
            c_x += x_c
        return Group(*all_r)

    def in_flanders_fields_the_poppies_blow(self):
        self.crosses_1 = self.crosses()[1]
        self.all_clouds = self.clouds(10)
        self.all_poppies = self.poppies()
        self.one = Group(self.sun(), self.field(), self.all_clouds, self.crosses_1, self.all_poppies, self.first_row,
                         self.bg())
        self.add(self.one)

    def between_the_crosses_row_on_row(self):
        self.trigger(4)
        self.play(
            self.camera.animate.scale_to(
                (2440, 800),
                (960, 540)
            ), time=2
        )
        self.trigger(7)
        self.play(
            self.camera.animate.scale_to(
                (0, 0),
                (self.canvas_width, self.canvas_height)
            ), time=2
        )

    def that_mark_our_place(self):
        """crosses fade to black and then to white, background turns red then transparant again"""
        self.trigger(9.5)

        for x in self.first_row.flattened_elements + self.second_row.flattened_elements:
            self.play(
                x.animate.color("black"),
                time=1
            )
            self.play(
                x.animate.color("white"),
                time=1.5
            )

        self.play(
            self.filter.animate.color([0, 0, 0, 0.5]),
            time=1
        )
        self.play(
            self.filter.animate.color([0, 0, 0, 0]),
            time=1.5
        )

    def and_in_the_sky(self):
        self.trigger(12.5)
        self.play(
            self.camera.animate.scale_to(
                (400, 400),
                (600, 500)
            ),
            self.filter.animate.color('#a5e5ff'),
            time=1
        )

    def the_larks_still_bravely_singing_fly(self):
        self.bird_svg = SVG(os.path.join(assets_dir, "skylark.svg"))
        notes = [[SVG(os.path.join(assets_dir, "note.svg")), (560, 500), (-60, -60), 'red'],
                 [SVG(os.path.join(assets_dir, "note.svg")), (540, 540), (-60, 0), 'white'],
                 [SVG(os.path.join(assets_dir, "note.svg")), (560, 580), (-60, 60), 'black']]
        for n in notes:
            note, color = n[0], n[-1]
            note.color = format_color(color, alpha=0)

        self.bird_svg.scale_to((700, 680), (380, 360))
        self.trigger(13.5)  # the lark
        self.add(self.bird_svg)

        note_init_size = (14, 40)

        for note, pos, move, color in notes: # notes (still bravely singing)
            self.trigger(15)
            self.add(note)  # still bravely singing
            note.scale_to(pos, note_init_size)
            self.play(
                note.animate.color(format_color(color, 1.)),
                time=1,
                ignore_multiple=True
            )
            self.trigger(15)
            self.play(
                note.animate.move(move), ignore_multiple=True,
                time=3
            )
            self.trigger(16)
            self.play(
                note.animate.color(format_color(color, 0.)), ignore_multiple=True,
                time=2
            )
            self.trigger(18)
            self.remove(note)

        self.trigger(18)
        self.play(
            self.bird_svg.animate.move((-600, -200))
        )
        self.trigger(19)
        self.remove(self.bird_svg)

    def scarce_heard_amid_the_guns_below(self):
        """adds blood side effect, along with paused field"""
        self.remove(self.crosses_1, self.first_row, self.all_poppies)
        self.trigger(19)
        self.remove(self.all_clouds)
        self.all_clouds_2 = self.clouds(10)
        self.add(self.all_clouds_2) # add new clouds
        self.remove(self.filter)
        self.bg_2()  # make new filter, resets self.filter
        self.add(self.filter)  # bring the filter to the front
        self.play(
            self.camera.animate.scale_to((0, 0), (self.canvas_width, self.canvas_height))
        )
        self.trigger(21)

        blood_shapes = [Circle((random.randint(0, self.canvas_width), random.randint(0, self.canvas_height)),
                               random.randint(400, 1000),
                               format_color((random.randint(100, 255), 0, 0), alpha=0))
                        for _ in range(100)]
        blood = Group(*blood_shapes)

        self.trigger(21.5)
        boom = Text(self.canvas_center, "BOOM", "Roboto", 100, weight="BOLD", color="white")
        boom.scale_to(self.canvas_center, (1500, 1000))
        self.add(boom)
        self.add(blood)
        self.play(
            self.filter.animate.color(format_color("black", alpha=1))
        )
        for smoke_c in blood_shapes:
            self.play(smoke_c.animate.color((*smoke_c.color[:-1], 0.9)))
        self.trigger(21.5)
        self.play(
            boom.animate.color(format_color("black", alpha=0))
        )
        self.trigger(22.5)
        self.remove(boom)
        # get rid of the blood
        for b in blood.flattened_elements:
            self.play(b.animate.move((0, random.uniform(self.canvas_height, self.canvas_height * 2))),
                      b.animate.color('black'))
        self.trigger(23.5)
        self.remove(blood)

    def we_are_the_dead(self):
        self.trigger(23.5)
        all_circles = []
        num_circles = 6000
        start = np.array([420, 220])
        self.init_radius = 20
        num_per_row = 100
        spacing = 30
        current_row = -1
        for c in range(num_circles):
            if c % num_per_row == 0:
                current_row += 1
            new_c = Circle(start + np.array([(c % num_per_row) * spacing, current_row * spacing]),
                           color='black',
                           size=self.init_radius)
            all_circles.append(new_c)

        self.all_circles = Group(*all_circles)
        self.add(self.all_circles)
        self.play(
            self.all_circles.animate.color(format_color('white', 1))
        )

    def transition_to_image(self, prev_circles, image_path, start_time, time, block_size, radius, center_offset):
        self.trigger(start_time)
        image = load_image(image_path)
        goal_circles, radius = generate_circles(self.canvas_width, self.canvas_height, image, block_size, radius)

        # print(len(goal_circles))
        prev_circle_elems = prev_circles.elements
        prev_circles_len = len(prev_circle_elems)
        is_rev = False
        goal_circles = sorted([[dist(self.canvas_center + center_offset, x.center), x] for x in goal_circles],
                                          key=lambda x: x[0], reverse=is_rev)
        prev_circle_elems_sorted = sorted([[dist(self.canvas_center, x.center), x] for x in prev_circle_elems], key=lambda x:x[0], reverse=is_rev)
        num_frames = time * FRAME_RATE
        to_add = []
        num_frames_iter = num_frames / len(goal_circles)
        self.trigger(start_time)
        for i, goal in enumerate(goal_circles):
            if i < prev_circles_len:
                c = prev_circle_elems_sorted[i][1]
            else:
                c = Circle(self.canvas_center, radius, color=format_color("white"))
                to_add.append(c)

            current_num_frames = round(num_frames_iter * i)
            self.play(c.animate.move_to(goal[1]), ignore_multiple=True, frames=current_num_frames)
            c.animate.offset = 0
            c.animate.wait_counter = 0
            self.play(c.animate.color(goal[1].color), ignore_multiple=True, frames=current_num_frames)

        prev_circles.add(*to_add)
        self.add(*to_add)

    def short_days_ago_we_lived(self):

        self.transition_to_image(self.all_circles,
                                 os.path.join(assets_dir, 'to_be_alive.png'),
                                 26,
                                 1.5,
                                 block_size=12,
                                 radius=10,
                                 center_offset=np.array([0, 0]))
        print('a done')

    def felt_dawn(self):
        self.transition_to_image(self.all_circles,
                                 os.path.join(assets_dir, 'sunrise.png'),
                                 28,
                                 1.5,
                                 block_size=12,
                                 radius=10,
                                 center_offset=np.array([-180, -100]))
        print('b done')

    def saw_sunset_glow(self):
        self.transition_to_image(self.all_circles,
                                 os.path.join(assets_dir, 'sunset.png'),
                                 30,
                                 1.5,
                                 block_size=12,
                                 radius=10,
                                 center_offset=np.array([-70, -190]))
        print('c done')

    def loved_and_were_loved(self):
        self.transition_to_image(self.all_circles,
                                 os.path.join(assets_dir, 'loved_2.png'),
                                 32.5,
                                 1.5,
                                 block_size=12,
                                 radius=10,
                                 center_offset=np.array([-70, -190]))
        print('d done')

    def and_now_we_lie_in_flanders_fields(self):
        self.trigger(35.2)
        for c in self.all_circles:
            self.play(c.animate.move((random.randint(-2000, 2000), random.randint(-2000, 2000))), ignore_multiple=True, time=1)
            c.animate.offset = 0
            c.animate.wait_counter = 0
            self.play(c.animate.color(format_color('black', 0)), ignore_multiple=True, time=1)

        self.play(
            self.filter.animate.color(format_color('black', 0))
        )

        self.add(self.crosses_1, self.first_row, self.all_poppies)
        self.trigger(36.2)
        self.remove(self.all_circles)

    def take_up_our_quarrel_with_the_foe(self):
        self.trigger(40)
        self.foe = SVG(os.path.join(assets_dir, "foe.svg"))
        self.foe.move_to(self.canvas_center)
        self.trigger(40.2)
        self.add(self.foe)
        self.play(self.filter.animate.color(format_color('black', 0.2)), time=0.4)
        self.trigger(40.6)
        self.play(
            self.foe.animate.scale_to(self.canvas_center-np.array([0, 4800]), self.foe.size*15), time=0.3
        )
        self.trigger(40.9)
        self.play(self.filter.animate.color(format_color('black', 1)), time=0.1)

    def to_you_from_failing_hands(self):
        self.trigger(41.5)
        self.remove(self.foe, self.crosses_1, self.first_row, self.all_poppies, self.all_clouds_2)
        skeleton_hand = SVG(os.path.join(assets_dir, "skeleton-hand.svg"))
        skeleton_hand.scale_to(self.canvas_center + np.array([0, 300]), skeleton_hand.size * 3)

        over_hand = Rectangle((skeleton_hand.leftmost, skeleton_hand.topmost), skeleton_hand.size, color='black')
        self.add(skeleton_hand, over_hand)
        self.play(
            over_hand.animate.color(format_color('black', 0)), time=0.5
        )
        self.trigger(42)

        pivot_point = np.array([skeleton_hand.center[0], skeleton_hand.bottommost])
        self.play(
            skeleton_hand.animate.rotate(pivot_point, 0, -45),
            time=0.5
        )
        self.play(
            skeleton_hand.animate.rotate(pivot_point, 0, 90)
        )

        self.play(
            skeleton_hand.animate.move((-5000, 0)),
            time=0.5
        )

    def we_throw_the_torch(self):
        flame_start = 43.5
        self.trigger(flame_start)

        flame_center = self.canvas_center + np.array([1500, 100])

        num = 4000
        r = 50
        flame_size_radius = 150
        flame_color_options = ['red', 'orange', 'yellow']
        flame_circles = []
        for _ in range(num):
            # p = random.randint(-rand_r, rand_r)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, flame_size_radius - r)
            x = flame_center[0] + distance * np.cos(angle)
            y = flame_center[1] + distance * np.sin(angle)

            flame_circles.append(
                Circle((x, y), r, random.choice(flame_color_options))
            )
        flame = Group(*flame_circles)
        self.add(flame)
        special_c = []
        for c in flame_circles:
            u = random.normalvariate(0, 1)
            if -2 < u < 2:
                self.play(
                    c.animate.move((-1500, random.randint(-300, 300))),
                    time=1,
                    ignore_multiple=True
                )
                special_c.append(c)

            else:
                t = random.uniform(0.5, 4)
                self.play(
                    c.animate.move((-6000, random.randint(-500, 500))),
                    time=t,
                    ignore_multiple=True
                )
                c.animate.offset = 0
                c.animate.wait_counter = 0
                self.play(
                    c.animate.color(format_color('yellow', 0)),
                    ignore_multiple=True,
                    time=min(t, 1.5)
                )

        catchup = 0.7
        div = int(len(special_c) / 2)
        before, after = special_c[:div], special_c[div:]
        for c in before:
            r = random.randint(0, 10) * 0.1

            self.current_frame = (flame_start + r) * FRAME_RATE
            c.animate.offset = 0
            c.animate.wait_counter = 0
            self.play(c.animate.move(
                (random.normalvariate(0, 100),
                 random.normalvariate(-750, 100))),
                time=catchup,
                ignore_multiple=True
            )
            c.animate.offset = 0
            c.animate.wait_counter = 0
            self.play(c.animate.color(format_color('yellow', 0)),
                      time=catchup,
                      ignore_multiple=True
                      )

        # delayed reaction
        self.trigger(flame_start+catchup)

        for c in after:
            t = random.uniform(0.5, 2)
            self.play(c.animate.move(
                                        (random.normalvariate(0, 300),
                                        random.normalvariate(-750, 300))),
                      time=t,
                      ignore_multiple=True
                      )
            c.animate.offset = 0
            c.animate.wait_counter = 0
            self.play(c.animate.color(format_color('yellow', 0)),
                      time=t,
                      ignore_multiple=True
                      )
        self.trigger(flame_start+catchup+2)
        for c in special_c:
            self.remove(c)

    def be_yours_to_hold_it_high(self):
        self.trigger(45)
        self.play(
            self.filter.animate.color(format_color('black', 0))
        )
        self.add(self.crosses_1, self.first_row, self.all_poppies)

    def if_ye_break_faith_with_us_who_die(self):
        # haphazard, destroyed greying flowers that tear and fade
        self.trigger(49)
        self.play(self.filter.animate.color(format_color('black', 0.4)))
        self.original_poppy_colors = []
        for p in self.all_poppies.flattened_elements:
            self.original_poppy_colors.append(np.array(p.color))
            self.play(
                p.animate.color('black')
            )

        self.play(
            self.sun_bg.color.animate.move_gradient_to(
                [1920, 1500, 100],
                [1920, 1500, 2400]
            ), time=2
        )
        self.play(
            self.filter.animate.color('black'),
            time=2
        )
        self.poppy_buds = [c.copy() for c in self.all_poppies.elements[1].elements]

    def we_shall_not_sleep(self):
        self.trigger(51)
        self.add(self.poppy_buds)
        filter_on_top = Rectangle((0,0), (self.canvas_width, self.canvas_height), color='black')
        self.add(filter_on_top)
        self.play(filter_on_top.animate.color(format_color('black', 0)), time=2)
        # reset animation for later
        for i, p in enumerate(self.all_poppies.flattened_elements):
            self.play(
                p.animate.color(self.original_poppy_colors[i]), time=0.1
            )

    def though_poppies_grow_in_flanders_fields(self):
        self.trigger(53)

        points = lambda a, o, r: np.array([o[0] + math.sin(r[0] * a) * r[1],
                                        o[1] - a])

        for p in self.poppy_buds:
            r = (random.uniform(0.004, 0.01), random.randint(200, 300))
            self.play(
                p.animate.move_along_func(partial(points, o=p.center, r=r), 0, random.randint(1000, 2000)),
                time=2,
                ignore_multiple=True
            )
            p.animate.offset = 0
            p.animate.wait_counter = 0
            self.play(p.animate.color(format_color('white', 0)),
                      time=2,
                      ignore_multiple=True
                      )

        self.play(
            self.filter.animate.color(format_color('white', 1)),
            time=1
        )
        self.play(
            self.filter.animate.color(format_color('black', 0)),
            time=2
        )

        self.play(
            self.sun_bg.color.animate.move_gradient_to(
                [1920, 500, 100],
                [1920, 500, 2400]
            ), time=3
        )

    def construct(self):
        # first stanza

        # in flanders fields the poppies blow
        self.in_flanders_fields_the_poppies_blow()
        self.between_the_crosses_row_on_row()
        self.that_mark_our_place()
        self.and_in_the_sky()
        self.the_larks_still_bravely_singing_fly()
        self.scarce_heard_amid_the_guns_below()
        # second stanza
        self.we_are_the_dead()
        self.short_days_ago_we_lived()
        self.felt_dawn()
        self.saw_sunset_glow()
        self.loved_and_were_loved()
        self.and_now_we_lie_in_flanders_fields()
        # third stanza
        self.take_up_our_quarrel_with_the_foe()
        self.to_you_from_failing_hands()
        self.we_throw_the_torch()

        self.be_yours_to_hold_it_high()
        self.if_ye_break_faith_with_us_who_die()
        self.we_shall_not_sleep()
        self.though_poppies_grow_in_flanders_fields()
        self.trigger(57)
        self.play(
            self.filter.animate.color('black')
        )


assets_dir = '/Users/juliancerniello/Documents/videos/flanders_fields/assets'
toy_example = Scene(background_color="white")

filename = 'flanders_fields.mp4'
toy_example.render(filename, start=0, time=60, audio=os.path.join(assets_dir, 'flanders_fields.wav'))

show_video(filename)
