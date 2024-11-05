from twiddler.canvas import *
from twiddler.shape import *
from twiddler.color.color import *
from twiddler.color.gradient import *
from twiddler.boolean_ops import *
from twiddler.helper import *
from twiddler.trackers import *
from twiddler.path import *
from sys import platform
import os


def show_video(filename):
    """
    shows the rendered video depending on the os type
    """
    try:
        file_path = os.path.join('video', filename)
        if platform == 'darwin':
            os.system(f"open {file_path}")
        else:
            os.system(os.path.abspath(file_path))
    except KeyboardInterrupt:
        pass
