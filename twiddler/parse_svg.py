import xml.etree.ElementTree as ET
from twiddler.path import Path
from twiddler.shape import *
from twiddler.helper import line_to_cubic_bezier
import svg.path as parser


# TODO: Use load_svg.py script to check for and correct any persistant errors. Some features aren't completely supported (gradients, use, defs, etc)
def point_to_list(point):
    """input: complex, output: list representing point in space"""
    return [point.real, point.imag]


def handle_transform(elem, transform_str):
    if not elem:
        return

    # transformations parser (everything but matrix)
    transformations = [['', '']]
    main_part = False
    for t in list(transform_str):
        if t == '(':
            main_part = True
            continue
        elif t == ')':
            transformations.append(['', ''])
            main_part = False
            continue

        if main_part:
            transformations[-1][1] += t
        else:
            transformations[-1][0] += t

    for t in list(range(len(transformations))[::-1]):  # do it in reverse so pop doesn't mess up order
        transformations[t][0] = transformations[t][0].strip()
        if ',' in transformations[t][1]:
            transformations[t][1] = [float(x) for x in transformations[t][1].replace(' ', '').split(',')]
        elif ' ' in transformations[t][1]:
            transformations[t][1] = [float(x) for x in transformations[t][1].strip().split(' ')]
        elif len(transformations[t][1].strip()):
            transformations[t][1] = [float(transformations[t][1])]
        else:
            transformations.pop(t)

    for t, x in transformations:
        if t == 'translate':
            if len(x) == 1:
                elem.move((x[0], 0))  # horizontal only
            else:
                elem.move(x)  # both or vertical only
        elif t == 'scale':
            if len(x) == 1:
                elem.scale(x[0], x[0]) # uniform scale
            else:
                elem.scale(*x)
        elif t == 'rotate':
            if len(x) == 1:
                elem.rotate((0, 0), x[0])  # rotation about origin
            else:
                elem.rotate((x[0], x[1]), x[2])  # rotation about defined point
        elif t == 'matrix':
            assert len(x) == 6
            a, b, c, d, e, f = x
            matrix = np.array([[a, c], [b, d]])
            if hasattr(elem, 'stroke_width'):
                if not elem.stroke_width:
                    elem.stroke_width = 1
                elem.stroke_width *= max(a, d)
            pos_change = np.array([e, f])
            full_matrix = np.identity(NUM_DIMS)
            matrix = np.array(matrix)
            full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
            elem.apply_matrix(lambda points: np.dot(points, full_matrix.T))
            elem.move(pos_change)
        else:
            raise Exception(t + " is an undefined transformation!")


def attr_or_default(attributes, name, default):
    if name in attributes:
        return attributes[name]
    return default


def remove_prefix(elem):
    """Remove "svg:" prefix from tags and attributes"""
    if '}' in elem.tag:
        prefix, elem.tag = elem.tag.split('}')
        prefix += '}'
        elem.attrib = {k.replace(prefix, ""): v for k, v in elem.attrib.items()}
        return prefix
    prefix = ''


def svg_to_path(e_attrib, style_attrs):
    d = parser.parse_path(attr_or_default(e_attrib, 'd', ''))

    nested_starts = []
    nested_points = [[]]
    last_was_move = False
    for i, x in enumerate(d):
        x_class = x.__class__

        if x_class == parser.Move:
            if not last_was_move:  # gets rid of the effects of multiple movements in a row
                if nested_starts:
                    nested_points.append([])
                nested_starts.append(point_to_list(x.end))
            else:
                # multiple movements, no points in between, just set the last move as the current one
                nested_starts[-1] = point_to_list(x.end)
            last_was_move = True
        else:
            last_was_move = False
            if x_class == parser.Line:
                nested_points[-1].extend(line_to_cubic_bezier(*point_to_list(x.start), *point_to_list(x.end)))
            elif x_class == parser.CubicBezier:
                control1, control2, end = point_to_list(x.control1), point_to_list(x.control2), point_to_list(x.end)
                nested_points[-1].extend([control1, control2, end])
            elif x_class == parser.QuadraticBezier:
                start, control, end = point_to_list(x.start), point_to_list(x.control), point_to_list(x.end)
                nested_points[-1].extend([start, control, end])  # has to be like cubic
            elif x_class == parser.Close:
                a, b = point_to_list(x.end), nested_starts[-1]
                nested_points[-1].extend(line_to_cubic_bezier(*a, *b))
                nested_points.append([])
            elif x_class == parser.Arc:
                arc_start = np.array(point_to_list(x.start))
                arc_end = np.array(point_to_list(x.end))
                arc_rotation = x.rotation
                large_arc = x.arc
                sweep = x.sweep

                assert x.radius.real == x.radius.imag
                radius = point_to_list(x.radius)
                arc = SVGArc(arc_start, radius, arc_rotation, large_arc, sweep, arc_end).points.squeeze()
                nested_points[-1].extend(arc)
            else:
                raise Exception('NOT IMPLEMENTED: ' + str(x_class))

    shape = Path(nested_points, nested_starts, **style_attrs)
    return shape


def svg_to_ellipse(e_attrib, style_kwargs):
    if 'r' in e_attrib:
        sx = e_attrib['r'] * 2
        sy = e_attrib['r'] * 2
    elif 'rx' in e_attrib:
        sx = e_attrib['rx'] * 2
        sy = e_attrib['ry'] * 2
    else:
        raise Exception('no radius defined!')
    has_cx, has_cy = 'cx' in e_attrib, 'cy' in e_attrib

    if has_cx and has_cy:
        cx, cy = float(e_attrib['cx']), float(e_attrib['cy'])
    elif has_cx:
        cx = float(e_attrib['cx'])
        cy = float(cx)
    elif has_cy:
        cy = float(e_attrib['cy'])
        cx = float(cy)
    else:
        raise Exception('Either cx or cy has to be defined!')

    ellipse = Ellipse((cx, cy),
                      (sx, sy),
                      **style_kwargs
                      )
    return ellipse


def svg_to_circle(e_attrib, style_kwargs):
    if 'r' in e_attrib:
        sx = e_attrib['r'] * 2
        sy = e_attrib['r'] * 2
    elif 'rx' in e_attrib:
        sx = e_attrib['rx'] * 2
        sy = e_attrib['ry'] * 2
    else:
        raise Exception('no radius defined!')
    circle = Circle((e_attrib['cx'], e_attrib['cy']),
                    (sx, sy),
                    **style_kwargs
                    )
    return circle


def svg_to_rect(e_attrib, style_kwargs):
    if 'x' in e_attrib:
        x = e_attrib['x']
    else:
        x = 0

    if 'y' in e_attrib:
        y = e_attrib['y']
    else:
        y = 0

    width, height = e_attrib['width'], e_attrib['height']
    rect = Rectangle((x, y), (width, height), **style_kwargs)
    return rect


def svg_to_text(e_attrib, style_kwargs):
    from twiddler.text import Text  # hacky, need to import here

    current_pos = np.array([e_attrib['x'], e_attrib['y']])
    if 'style' in e_attrib:
        text_style_attrs = get_attrs_from_str(e_attrib['style'])
    else:
        text_style_attrs = {}

    for x in [['font_family', 'DEFAULT'], ['font_size', 12], ['line_spacing', 'NORMAL'], ['font_weight', 'NORMAL'], ['line_height', 'NORMAL']]:
        if x[0] not in text_style_attrs:
            text_style_attrs[x[0]] = x[1]

    text = Text(current_pos,
                e_attrib['text'],
                text_style_attrs['font_family'].replace('\'', '').replace('\"', ''),
                font_size=text_style_attrs['font_size'],
                line_spacing=text_style_attrs['line_height'],
                weight=text_style_attrs['font_weight'].upper() if isinstance(text_style_attrs['font_weight'], str) else 'NORMAL',
                from_svg=True,
                use_cache=False,
                **style_kwargs
                )
    return text


def get_attrs_from_str(attrs):
    attrs_sep = [a.split(':') for a in attrs.split(';') if (':' in a and len(a.split(':')) == 2)]
    attrs = {k.strip(): v.strip() for k, v in attrs_sep}
    return attrs


def evaluate(k, v, dict_to_edit):
    """changes a style attribute if it qualifies"""
    k = k.strip().lower()
    is_none = ['none', 'None']

    if isinstance(v, str):
        v = v.strip()
        if v[-2:] == 'px':
            v = v[:-2]

    if hasattr(v, 'replace') and v.replace('.', '', 1).replace('-', '', 1).isdigit():
        v = float(v)

    if v in is_none:
        v = None

    if v is not None:
        if k == 'fill':
            dict_to_edit['fill'] = True
            dict_to_edit['color'] = v
        elif k == 'stroke':
            dict_to_edit['stroke'] = True
            dict_to_edit['stroke_color'] = v
            if not dict_to_edit['stroke_width']:
                dict_to_edit['stroke_width'] = 1  # set default stroke value if not set
        elif k in ['stroke-width', 'fill-opacity', 'stroke-opacity']:
            dict_to_edit[k.replace('-', '_')] = v
        elif k == 'opacity':
            dict_to_edit['fill_opacity'] = v
            if dict_to_edit['stroke_opacity'] == 1:  # 1 is the default
                dict_to_edit['stroke_opacity'] = v
        elif k == 'stroke-linecap':
            dict_to_edit['line_cap'] = v.upper()
        elif k == 'stroke-linejoin':
            dict_to_edit['line_join'] = v.upper()
        elif k == 'fill-rule':
            dict_to_edit['fill_rule'] = v.upper()


def format_main_dim(dim_type, root, default_dim):
    dim = root.attrib[dim_type].replace('px', '')
    if '%' in dim:
        dim = float(dim.replace('%', '')) / 100 * default_dim
    else:
        dim = float(dim)
    return dim


class SVG(Group):
    def __init__(self, path, user_default_size=None, **kwargs):
        super().__init__(**kwargs)
        svg_data = open(path, "r").read()
        self.root = ET.fromstring(svg_data)
        self.prefix = remove_prefix(self.root)
        self.id_and_class_style_attributes = {}
        self.hrefs = {}
        self.handle_style()
        disallow = ['width', 'height', 'viewBox', 'version', 'xmlns', 'xmlns:xlink']
        root_attrib = self.root.attrib
        self.parse(self.root, self, {x: root_attrib[x] for x in root_attrib if x not in disallow})
        # rescale based on viewBox
        default_dim = 300
        if 'width' in self.root.attrib and 'height' in self.root.attrib:
            main_width = format_main_dim('width', self.root, default_dim)
            main_height = format_main_dim('height', self.root, default_dim)
        else:
            main_width, main_height = self.size if not user_default_size else user_default_size

        if 'viewBox' in self.root.attrib:
            view_box = self.root.attrib['viewBox'].strip()
            if view_box and ' ' in view_box:
                view_box = [float(x) for x in view_box.split(' ')]
                assert len(view_box) == 4
                min_x, min_y, view_width, view_height = view_box
                self.scale(main_width / view_width, main_height / view_height)

    def handle_style(self):
        """custom svg parser"""

        style_tags = self.root.findall('.//{}style'.format(self.prefix))
        # simple css parser
        for s in style_tags:
            is_meat = False
            current = ''
            names = []
            for c in list(s.text):
                if c == '{':
                    is_meat = True
                elif c == '}':
                    is_meat = False
                    if current:
                        if current.strip()[-1] != ';':
                            current = current.strip() + ';'

                        self.id_and_class_style_attributes[names[-1]] += current

                        current = ''
                    continue

                if is_meat:
                    if current and c == '{':
                        current = current.strip()
                        assert len(current) > 0
                        names.append(current)
                        if current not in self.id_and_class_style_attributes:
                            self.id_and_class_style_attributes[current] = ''
                        current = ''
                    else:
                        current += c
                else:
                    current += c

        # some styles have more than one class/id
        for c in list(self.id_and_class_style_attributes.keys()):
            c = c.replace(' ', '')
            if ',' in c:
                original_c = str(c)
                c = c.split(',')
                for item in c:
                    if item not in self.id_and_class_style_attributes:
                        self.id_and_class_style_attributes[item] = ''

                    self.id_and_class_style_attributes[item] += self.id_and_class_style_attributes[original_c]
                del self.id_and_class_style_attributes[original_c]

    def parse(self, source, parent, parent_attrs=None):
        for e in source:
            add_elem = True
            remove_prefix(e)
            new_elem = None
            e_tag = e.tag
            e_attrib = parent_attrs.copy() if parent_attrs else {}  # copy parent attributes

            for a in e.attrib:
                current_attr = e.attrib[a]
                if isinstance(current_attr, str) and current_attr.replace('.', '').replace('-', '').strip().isdigit():
                    e_attrib[a] = float(current_attr)
                else:
                    if a == 'transform' and a in e_attrib:  # attributes which add onto each other
                        e_attrib[a] = current_attr + ' ' + e_attrib[a]
                    else:
                        e_attrib[a] = current_attr
            e_style_kwargs = self.set_style_kwargs(e, e_attrib)
            if e_tag == 'defs':
                new_elem = Group()
                self.parse(e, new_elem, e_attrib)
                add_elem = False # assume that this is bening referenced somewhere (in hrefs, for example)
            elif e_tag == 'g':
                new_elem = Group()
                self.parse(e, new_elem, e_attrib)  # recursive adding, has structure
                if len(new_elem.flattened_elements) == 0:
                    del new_elem  # delete the group if it is empty
                    new_elem = None
            else:
                if e_tag == 'path':
                    assert 'd' in e_attrib
                    new_elem = svg_to_path(e_attrib, e_style_kwargs)
                elif e_tag == 'rect':
                    new_elem = svg_to_rect(e_attrib, e_style_kwargs)
                elif e_tag == 'line':
                    d = 'M {},{} L {},{}'.format(e_attrib['x1'], e_attrib['y1'], e_attrib['x2'], e_attrib['y2'])
                    e_attrib['d'] = d
                    new_elem = svg_to_path(e_attrib, e_style_kwargs)
                elif e_tag == 'circle':
                    new_elem = svg_to_circle(e_attrib, e_style_kwargs)
                elif e_tag == 'ellipse':
                    new_elem = svg_to_ellipse(e_attrib, e_style_kwargs)
                elif e_tag in ['polygon', 'polyline']:
                    assert 'points' in e_attrib
                    # get the first point (without parsing, that is done later)
                    points = e_attrib['points']

                    e_attrib['d'] = 'M' + points
                    if e_tag == 'polygon':  # not polyline, polyline stays open
                        e_attrib['d'] += 'Z'
                    new_elem = svg_to_path(e_attrib, e_style_kwargs)
                elif e_tag == 'text':
                    # not yet supported!!!
                    # check for children (tspan)
                    if len(e):
                        additional_text = '\n'.join([x.text for x in e])
                        e_attrib['text'] = additional_text
                    elif 'text' not in e_attrib:
                        e_attrib['text'] = e.text

                    new_elem = svg_to_text(e_attrib, e_style_kwargs)
                elif e_tag == 'use':
                    # see https://developer.mozilla.org/en-US/docs/Web/SVG/Element/use
                    href = e.attrib['href']
                    if href in self.hrefs:
                        elems_to_reference = [x.attrib['elem'].copy() for x in self.hrefs[href]]
                        new_elem = Group(*elems_to_reference)

                        use_attr_x, use_attr_y = 0., 0.
                        if 'x' in e.attrib:
                            use_attr_x = float(e.attrib['x'].replace('px', ''))
                        if 'y' in e.attrib:
                            use_attr_y = float(e.attrib['y'].replace('px', ''))
                        new_elem.move((use_attr_x, use_attr_y))

                if 'transform' in e_attrib:
                    handle_transform(new_elem, e_attrib['transform'])

            if new_elem is not None and add_elem:
                e.attrib['elem'] = new_elem  # save the rendered elements to the xml base (this is so "use" works correctly)
                parent.add(new_elem)

    def set_style_kwargs(self, e, attrs):
        """sets the style attributes"""
        e_tag = e.tag
        if 'style' not in attrs:
            attrs['style'] = ''
        if e_tag in self.id_and_class_style_attributes:
            attrs['style'] += self.id_and_class_style_attributes[e_tag]
        if 'class' in attrs:
            classes = attrs['class'].strip().split(' ')
            for c in classes:
                c = '.' + c
                if c in self.id_and_class_style_attributes:
                    attrs['style'] += self.id_and_class_style_attributes[c]
                if e_tag not in ['use', 'g']: # if group, it propagates ot the individual elements
                    if c not in self.hrefs:  # add a reference in case it's needed (use tag)
                        self.hrefs[c] = []
                    self.hrefs[c].append(e)
        if 'id' in attrs:
            ids = attrs['id'].split(' ')
            for item in ids:
                item = '#' + item
                if item in self.id_and_class_style_attributes:
                    attrs['style'] += self.id_and_class_style_attributes[item]
                if e_tag not in ['use', 'g']: # if group, it propagates ot the individual elements
                    if item not in self.hrefs:  # add a reference in case it's needed (use tag)
                        self.hrefs[item] = []
                    self.hrefs[item].append(e)
        style_vals = {
            'fill': False,
            'color': COLOR_DEFAULT,
            'stroke': False,
            'stroke_color': COLOR_DEFAULT,
            'stroke_width': None,
            'fill_opacity': 1,
            'stroke_opacity': 1,
            'fill_rule': 'NONZERO',
            'line_cap': 'BUTT',
            'line_join': 'MITER'
        }
        if 'style' in attrs:  # let this run first (before the additional calls), to overwrite existing values
            extra_style_attrs = get_attrs_from_str(attrs['style'])
            for e in extra_style_attrs:
                attrs[e] = extra_style_attrs[e]
            del attrs['style']

        for k, v in attrs.items():
            evaluate(k, v, style_vals)

        if e_tag == 'path':  # additional rules for paths
            if style_vals['fill'] is False and style_vals['stroke'] is False and 'fill' not in attrs:
                style_vals['fill'] = True
        return style_vals