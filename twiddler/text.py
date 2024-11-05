from manimpango import MarkupUtils
import manimpango
from twiddler.constants import *
from contextlib import contextmanager
from pathlib import Path
import os
from twiddler.parse_svg import SVG
from twiddler.helper import make_new_directories
import hashlib
START_X = 30
START_Y = 20


class Text(SVG):
    def __init__(self, start, text, font, font_size, line_spacing=1, slant='NORMAL', weight='NORMAL', disable_ligatures=False,
                 justify=None, color=COLOR_DEFAULT, file_dir=None, use_cache=False, line_width=2000, center=False, from_svg=False, offset=(5, 0), **kwargs):
        if not file_dir:
            file_dir = os.path.join(VIDEO_PATH, STATIC_FILES, SVG_FILES)
        self.font_scalar = 100
        self.x, self.y = start
        self.text = str(text)
        self.font = font
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.slant = slant
        self.weight = weight
        self.disable_ligatures = disable_ligatures
        self.justify = justify
        self.text_color = color
        self.line_width = line_width
        self.center_align = center
        file_name = self.settings_to_hash() + '.svg'
        make_new_directories(file_dir)
        self.file_path = os.path.join(file_dir, file_name)
        path_exists = os.path.exists(self.file_path)
        if not path_exists or (path_exists and not use_cache):
            self.render_text_svg()
        super().__init__(self.file_path, **kwargs)
        if not from_svg:
            to_scale = self.font_size / self.font_scalar * 0.75
            self.scale(to_scale, to_scale)
            self.move_to((self.x + offset[0] + self.height/1.5, self.y + offset[1]), self.corner, center=False)

    @property
    def height(self):
        all_bottoms = []
        all_tops = []

        for elem in self.elements:
            if elem.bottommost:
                all_bottoms.append(elem.bottommost)
            if elem.topmost:
                all_tops.append(elem.topmost)
        height = max(all_bottoms) - min(all_tops)
        return height

    @property
    def corner(self):
        i = 0
        left = self.elements[0].leftmost
        while not left:
            if i < len(self.elements) - 1:
                i += 1
                left = self.elements[i].leftmost

        all_tops = []
        for elem in self.elements:
            if elem.topmost:
                all_tops.append(elem.topmost)

        return left, min(all_tops)

    def settings_to_hash(self):
        settings = (
                "PANGO" + self.font + self.slant + self.weight
        )
        settings += str(self.line_spacing) + str(self.font_size)
        settings += str(self.disable_ligatures)
        id_str = self.text + settings
        hasher = hashlib.sha256()
        hasher.update(id_str.encode())
        return hasher.hexdigest()[:16]

    def render_text_svg(self):
        if self.text_color:
            self.text = '<span foreground="{}">{}</span>'.format(self.text_color, self.text)

        validate_error = MarkupUtils.validate(self.text)
        if validate_error:
            raise ValueError('Text value error occurred')

        MarkupUtils.text2svg(
            self.text,
            self.font,
            self.slant,
            self.weight,
            self.font_scalar,
            self.line_spacing,
            self.disable_ligatures,
            self.file_path,
            START_X,
            START_Y,
            2000,
            2000,
            justify=self.justify,
            pango_width=self.line_width,
        )


@contextmanager
def register_font(font_file: Path):
    font_file = os.fspath(font_file)
    init = manimpango.list_fonts()
    assert manimpango.register_font(font_file), "Invalid Font possibly."
    final = manimpango.list_fonts()
    yield list(set(final) - set(init))[0]
    assert manimpango.unregister_font(font_file), "Can't unregister Font"

