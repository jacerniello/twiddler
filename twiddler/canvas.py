import cairo
import os
import ffmpeg
from tqdm import tqdm
from twiddler.shape import *
from twiddler.helper import divide_chunks, flatten_list
from twiddler.group import Group
from twiddler.parse_svg import SVG
from twiddler.text import Text
from twiddler.color.gradient import LinearGradient, RadialGradient
from twiddler.camera.camera import Camera
from twiddler.animate.camera_animate import CameraAnimation
from twiddler.path import Path


class Config:
    def __init__(self, canvas_width, canvas_height):
        self.main = {
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }


class BasicCanvasConfig(Config):
    """used to configure the canvas object with values, which can be changed from their default"""
    def __init__(self, canvas_width=1920, canvas_height=1080):
        super().__init__(canvas_width, canvas_height)


class FourKCanvasConfig(Config):
    """used to configure the canvas object with values, which can be changed from their default"""
    def __init__(self, canvas_width=3840, canvas_height=2160):
        super().__init__(canvas_width, canvas_height)


class ShortsConfig(Config):
    """used to configure the canvas object with YouTube Shorts resolution"""
    def __init__(self, canvas_width=1080, canvas_height=1920):
        super().__init__(canvas_width, canvas_height)


class Canvas:
    """handles the scene and all of its moving parts.
    Includes the function to render a scene"""
    def __init__(self, background_color=COLOR_DEFAULT, config=None):
        # reads the config values
        self.config = config if config else FourKCanvasConfig()
        self.canvas_format = cairo.FORMAT_ARGB32
        self.canvas_width = self.config.main['canvas_width']
        self.canvas_height = self.config.main['canvas_height']
        self.alpha = 0
        self.start_frame = 0  # the frame to start the video at; allows for skipping around
        self.current_frame = 0  # the current video frame
        self.video_started = False
        self.elements = []
        self.animations = {}
        self.max_frame_val = 0
        self.initial_bg_color = background_color  # default background color unless otherwise specified
        self.surface = cairo.ImageSurface(self.canvas_format,
                                          self.canvas_width,
                                          self.canvas_height)
        self.camera = Camera(cairo.Context(self.surface),
                             width=self.canvas_width,
                             height=self.canvas_height)

    @property
    def canvas_center(self):
        """gets the center of the canvas, dynamically"""
        return np.array([self.canvas_width / 2, self.canvas_height / 2])

    def construct(self):
        """meant to be edited by the user when they do class inheritance"""
        return

    def trigger(self, time_in):
        """use trigger to change the current frame; this should be used to define the start and stop for elements"""
        self.current_frame = time_in * FRAME_RATE

        # reset the offset of all animations to make them start at designated time
        # resets the wait_counter because now there is no reason to wait for an animation to stop (we're setting the start regardless)
        for f in self.animations:
            for animation_obj in self.animations[f]:
                animation_obj.animate.offset = 0
                animation_obj.animate.wait_counter = 0

    def add(self, *elements):
        """add shape object to be drawn; this runs for each shape where add has been called"""
        if not self.video_started:  # if the video has not started yet, add the elements
            # elements--including their start and stop--are defined by the user at the beginning of processing, so this only needs to run once
            # once added to the list of elements to render, this is irrelevant
            for elem in elements:
                if isinstance(elem, list):
                    # recursion in case of list
                    self.add(*elem)
                    continue
                elif hasattr(elem, "elements") and not hasattr(elem, "is_boolean_op"):  # so like a Group, Text, SVG
                    for g in elem.elements:
                        self.add(g)
                    continue

                elem.start_end.append([self.current_frame, math.inf])  # we do not know the stop yet, math.inf is a placeholder
                if elem not in self.elements:
                    # add the element to be rendered if it hasn't been already
                    self.elements.append(elem)
                elem.canvas = self

    def remove(self, *elements):
        # remove shape elements
        if not self.video_started:  # only run from the construct method, before the video has started rendering
            for elem in elements:
                if isinstance(elem, list):
                    self.remove(elem)
                    continue
                elif hasattr(elem, "elements") and not hasattr(elem, "is_boolean_op"):
                    for g in elem.elements:
                        self.remove(g)
                    continue

                next_end = self.current_frame  # frame to remove the element; set by trigger
                # print('remove', next_end)
                if len(elem.start_end) == 0:
                    # add() was not called for this element; the start_end list is empty
                    raise Exception('Cannot remove an element that has not been added!')
                else:
                    if len(elem.start_end[-1]) == 2:  # to ensure something didn't go horribly wrong
                        elem.start_end[-1][1] = next_end  # sets the end time with the previous start time
                    else:
                        # case handling: if the previous start_end is invalid length (i.e. too big)
                        raise Exception("Length invalid for elem")

    def set_color(self, color, do_format=True, alpha=None, canvas=None):
        if not canvas:
            canvas = self.camera.canvas  # defines the canvas that we are making changes to
        if do_format:
            color = format_color(color)  # format the color if it hasn't been formatted already
        if isinstance(color, (LinearGradient, RadialGradient)):
            canvas.set_source(color.source())  # if the color is a gradient, set the source to that color's source
        elif len(color) == 4:
            # if the color is rgba
            canvas.set_source_rgba(*color)
        else:
            if not alpha:
                alpha = 1
            canvas.set_source_rgba(*color, float(alpha))  # sets the color as rgba even if a is not defined

    def specific_draw(self, start_point, points, as_bezier, close_path):
        """drawing specific for shape-like elements"""
        self.camera.canvas.move_to(*start_point)
        if as_bezier:
            # draws points as bezier curve(s)
            chunks = divide_chunks(points, 3)  # <- for cubic bezier, default
            for chunk in chunks:
                flattened_chunk = flatten_list(chunk)
                self.camera.canvas.curve_to(*flattened_chunk)
        else:
            # draw points as line(s)
            for point in points:
                self.camera.canvas.line_to(*point)

        if close_path:  # closes the path if specified, otherwise, leaves open
            self.camera.canvas.close_path()

    def draw(self, elem):
        """draws the element using coordinates"""
        if isinstance(elem, (Group, SVG, Text)):
            # recursion for groups of elements
            for g_elem in elem.elements:
                self.draw(g_elem)
        else:
            # boolean variables
            stroke = elem.stroke
            fill = elem.fill
            if not stroke and not fill:  # only draw if one of these is specified
                return

            if elem.fill_rule:

                fill_rule_upper = elem.fill_rule
                if fill_rule_upper == 'NONZERO':
                    fill_rule = cairo.FILL_RULE_WINDING
                elif fill_rule_upper == 'EVENODD':
                    fill_rule = cairo.FILL_RULE_EVEN_ODD
                else:
                    raise Exception(elem.fill_rule, "is an unsupported fill rule!")
                self.camera.canvas.set_fill_rule(fill_rule)

            if isinstance(elem, Path):
                for i, x in enumerate(elem.points):
                    self.specific_draw(elem.start[i], x, elem.as_bezier, elem.close_paths[i])
            else:
                self.specific_draw(elem.start, elem.points, elem.as_bezier, elem.close_path)

            if elem.line_cap:
                line_cap_upper = elem.line_cap.upper().strip()

                if line_cap_upper == 'ROUND':
                    line_cap = cairo.LINE_CAP_ROUND
                elif line_cap_upper == 'SQUARE':
                    line_cap = cairo.LINE_CAP_SQUARE
                else:
                    line_cap = cairo.LINE_CAP_BUTT
                self.camera.canvas.set_line_cap(line_cap)

            if elem.line_join:
                line_join_upper = elem.line_join.upper().strip()

                if line_join_upper == 'ROUND':
                    line_join = cairo.LINE_JOIN_ROUND
                elif line_join_upper == 'BEVEL':
                    line_join = cairo.LINE_JOIN_BEVEL
                else:
                    line_join = cairo.LINE_JOIN_MITER
                self.camera.canvas.set_line_join(line_join)

            if elem.stroke_width:
                self.camera.canvas.set_line_width(elem.stroke_width)

            if fill and stroke:
                # stroking and filling (filling is slightly different in this case)
                self.set_color(elem.color)
                self.camera.canvas.fill_preserve()
                self.set_color(elem.stroke_color)
                self.camera.canvas.stroke()
            elif fill:
                # just filling
                self.set_color(elem.color)
                self.camera.canvas.fill()
            elif stroke:
                # just stroking
                self.set_color(elem.stroke_color)
                self.camera.canvas.stroke()

    def reset_frame(self):
        """defines the background by setting it as a rectangle; resets the frame"""
        if not self.video_started:
            self.background = Rectangle((0, 0), (self.canvas_width, self.canvas_height), color=self.initial_bg_color)
            self.add(self.background)

    def setup_single_frame(self, start=0):
        """for the rendering of a single frame, given the start time"""
        if start is None:
            start = 0
        self.start_frame = int(start * FRAME_RATE)
        self.init_video()
        for f in self.animations:
            for animation in self.animations[f]:
                self.play(animation)

    def unpack_elem_changes(self, elem):
        """gets the element's most recent start and an end (based on video start)"""
        if self.start_frame != 0:
            while elem.start_end and len(elem.start_end[0]) == 2 and elem.start_end[0][1] < self.start_frame:
                # basically, remove all changes that have already passed. This is when we don't start at the beginning of the video like normal
                elem.start_end.pop(0)

        if len(elem.start_end) == 0:
            return False

        changes = elem.start_end[0]
        return changes

    def init_animations(self, num_frames):
        for f in range(num_frames):
            self.animations[f] = []
        self.max_frame_val = num_frames

    def render_frame(self, file_path=None, start=None, time=None):
        """makes the cairo surface and canvas, which can be written to"""
        self.reset_frame()
        if len(self.animations) == 0:
            self.init_animations(int(time * FRAME_RATE))

        if start is not None or time is not None:
            # renders only a specific frame
            self.setup_single_frame(start)

        offset = 0
        for j in range(len(self.elements)):
            elem = self.elements[j - offset]
            out = self.unpack_elem_changes(elem)
            if out:
                elem_start, elem_end = out
            else:
                # removes the element from future consideration if it doesn't have aligning events
                self.elements.pop(j - offset)
                offset += 1
                continue
            self.camera.canvas.save()
            if self.current_frame >= elem_start:
                if self.current_frame < elem_end:
                    # draws the element if it is within bounds
                    self.draw(elem)
                else:
                    # current_frame >= elem_end, out of the bounds
                    if len(elem.start_end) > 1:
                        elem.start_end.pop(0)  # if the element has other animations(start_end times), only remove the completed one
                        # DO NOT REMOVE THE ELEMENT ITSELF FROM THE elements LIST
                    else:
                        # removes the element from the elements list, adds offset to counteract issues with pop and reference the correct elements
                        self.elements.pop(j - offset)
                        offset += 1

            self.camera.canvas.restore()
        if file_path:
            # for single frame, adds path to save
            file_path = os.path.join(VIDEO_PATH, file_path)
            self.surface.write_to_png(file_path)
        buf = self.surface.get_data()  # otherwise, get the rendered frame data for the video
        return buf

    def run_animation(self, animation_obj):
        """runs the animation"""
        if not hasattr(animation_obj, "start") or not animation_obj.start:
            # starts the animation; initializes the corresponding variables to do so
            animation_obj.start = True
            animation_obj.animate.num_running += 1  # adds 1 to animate manager
        animation_obj.animation_func(canvas=self, **animation_obj.kwargs)
        animation_obj.animation_frames_left -= 1

    def play(self, *all_animations, ignore_multiple=True, time=None, frames=None):
        if not self.video_started:
            # add initial animations that are set using the construct method
            for animation_obj in all_animations:
                if time:  # duration that the animation should run
                    animation_obj.set_animation_time(time)
                if frames:
                    animation_obj.set_animation_time_from_num_frames(frames) # alternatively, set # of frames to run for

                if animation_obj not in self.animations[animation_obj.first_frame]:
                    animation_obj.first_frame = self.current_frame + animation_obj.animate.wait_counter + 1  # starts the frame after the shape has been rendered (doesn't start immediately)
                    # animation_obj.animate.offset accounts for currently running animations
                    # trigger resets this offset so that any event occurs exactly at that point in time.
                    # this is especially useful for defining when a sequence of events should begin
                    # without this offset all events for a single animation_obj would run at the same time, which is not the ideal use case
                    animation_obj.first_frame += animation_obj.animate.offset
                    animation_obj.animate.offset += animation_obj.animation_frames  # offset by the previous frames already designated toward an animation
                    animation_obj.last_frame = animation_obj.first_frame + animation_obj.animation_frames - 1  # offset to account for the + 1 in first_frame, to stop at the right time
                    animation_obj.animate.wait_counter += 1  # time to wait to play the next animation for the given element; adds a single frame between start and stop
                    animation_obj.animation_frames_left = animation_obj.animation_frames
                    # print('z', animation_obj.first_frame, animation_obj.last_frame, animation_obj.last_frame-animation_obj.first_frame, animation_obj.animation_frames_left)
                    if animation_obj.first_frame <= self.max_frame_val:   # make sure that the frame is actually in the video
                        self.animations[animation_obj.first_frame].insert(0, animation_obj)  # insert the animation into correct frame; found to be important based on testing
        else:
            all_animations = [x for x in all_animations if x]  # removes empty values
            # run existing animations as given, such as in the render method
            remove_any = False
            for animation_obj in all_animations:
                # print(animation_obj.animate.num_running, animation_obj.animate, self.current_frame)
                if animation_obj.animate.num_running > 1 and not ignore_multiple:
                    raise Exception(animation_obj, "is running more than one animation at once. This type of behavior is not allowed.")
                if self.current_frame >= animation_obj.last_frame:  # animation should be stopped on last_frame, = is just a failsafe
                    # print('running', animation_obj.animation_frames_left)
                    # print(animation_obj.first_frame, animation_obj.animation_frames_left,  animation_obj.animation_frames)
                    if animation_obj.animation_frames_left == animation_obj.animation_frames:
                        # print("tttt")
                        # animation is finished but was never run; must be because it was skipped (using start). Display as finished
                        animation_obj.animation_start_pos = animation_obj.animation_frames
                    self.run_animation(animation_obj)
                    animation_obj.removed = True
                    animation_obj.animate.num_running -= 1  # removes 1 from manager for running animations
                    remove_any = True
                elif self.current_frame >= animation_obj.first_frame:  # otherwise, animation should still keep running
                    if animation_obj.animation_start_pos is None:
                        # print(self.current_frame, animation_obj.first_frame)
                        animation_obj.animation_start_pos = self.current_frame - animation_obj.first_frame + 1
                        # print('start:', animation_obj.animation_start_pos)
                    self.run_animation(animation_obj)

            if remove_any:
                for t in self.animations:
                    self.animations[t] = [x for x in self.animations[t] if not x.removed]  # gets rid of removed animations

    def init_video(self):
        """starts the video"""
        self.reset_frame()
        self.construct()  # will have been modified via the program
        self.video_started = True
        self.current_frame = self.start_frame

    def render(self, render_path, time=None, start=0, stop=None, num_frames=None, audio=None, output=True):
        if not os.path.exists(VIDEO_PATH):
            os.mkdir(VIDEO_PATH)

        if not num_frames:
            if not time:
                raise Exception('Either time or num_frames needs to be defined')
            num_frames = int(time * FRAME_RATE)

        if len(self.animations) == 0:
            self.init_animations(int(time * FRAME_RATE))

        if start:
            self.start_frame = int(start * FRAME_RATE)

        if stop:
            to_stop = int(stop * FRAME_RATE)
        else:
            to_stop = num_frames
        if output:
            file_path = os.path.join(VIDEO_PATH, render_path)
            self.file_path = file_path
            if os.path.exists(file_path):
                # resets the file path for the video
                os.remove(file_path)
            if audio:
                # adds audio at given time if audio is provided
                audio_output = ffmpeg.input(audio, ss=start).audio.filter('atrim', duration=stop-start if stop else time-start)
                video_output = ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb32",
                    s="{}x{}".format(self.canvas_width, self.canvas_height),
                    framerate=FRAME_RATE,
                )

                process = (
                    ffmpeg.concat(video_output, audio_output, v=1, a=1)
                    .output(file_path, pix_fmt='yuv420p', crf=18, preset='slow',
                            loglevel='quiet',
                            color_primaries='bt709', color_trc='bt709', colorspace='bt709')
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            else:
                # otherwise, just add video
                process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', framerate=FRAME_RATE, pix_fmt='rgb32',
                           s='{}x{}'.format(self.canvas_width, self.canvas_height))
                    .output(file_path, pix_fmt='yuv420p', crf=18, preset='slow',
                            loglevel='quiet',
                            color_primaries='bt709', color_trc='bt709', colorspace='bt709')
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )

            self.init_video()  # set up the video
            for _ in tqdm(range(self.start_frame, to_stop)):
                for t in self.animations: self.play(*self.animations[t])  # play the valid animations
                buf = self.render_frame()  # gets the rendered data for the current frame
                self.current_frame += 1  # sets up to get the next frame
                # converts the frame data to np array and then writes it to the video file
                frame_data = np.ndarray(shape=(self.canvas_height, self.canvas_width),
                                        dtype=np.uint32,
                                        buffer=buf)

                process.stdin.write(
                    frame_data.tobytes())
            process.stdin.close()
            process.wait()
        else:
            self.init_video()  # set up the video
            for _ in tqdm(range(self.start_frame, to_stop)):
                for t in self.animations: self.play(*self.animations[t])  # play the valid animations
                self.current_frame += 1
