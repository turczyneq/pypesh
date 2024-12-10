"""
Custom fancy arrowstyle
"""

import matplotlib.patches
import functools

_Style = matplotlib.patches._Style

from matplotlib.bezier import (
    NonIntersectingPathException,
    get_cos_sin,
    get_intersection,
    get_parallels,
    inside_circle,
    make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath,
    split_path_inout,
)
from matplotlib.path import Path
import numpy as np


def _point_along_a_line(x0, y0, x1, y1, d):
    """
    Return the point on the line connecting (*x0*, *y0*) -- (*x1*, *y1*) whose
    distance from (*x0*, *y0*) is *d*.
    """
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy) ** 0.5
    x2, y2 = x0 - ff * dx, y0 - ff * dy

    return x2, y2


class FenixArrowStyle(_Style):
    _style_list = {}

    class _Base:

        @staticmethod
        def ensure_quadratic_bezier(path):

            segments = list(path.iter_segments())
            if (
                len(segments) != 2
                or segments[0][1] != Path.MOVETO
                or segments[1][1] != Path.CURVE3
            ):
                raise ValueError("'path' is not a valid quadratic Bezier curve")
            return [*segments[0][0], *segments[1][0]]

        def transmute(self, path, mutation_size, linewidth):

            raise NotImplementedError("Derived must override")

        def __call__(self, path, mutation_size, linewidth, aspect_ratio=1.0):
            """
            The __call__ method is a thin wrapper around the transmute method
            and takes care of the aspect ratio.
            """

            if aspect_ratio is not None:
                # Squeeze the given height by the aspect_ratio
                vertices = path.vertices / [1, aspect_ratio]
                path_shrunk = Path(vertices, path.codes)
                # call transmute method with squeezed height.
                path_mutated, fillable = self.transmute(
                    path_shrunk, mutation_size, linewidth
                )
                if np.iterable(fillable):
                    # Restore the height
                    path_list = [
                        Path(p.vertices * [1, aspect_ratio], p.codes)
                        for p in path_mutated
                    ]
                    return path_list, fillable
                else:
                    return path_mutated, fillable
            else:
                return self.transmute(path, mutation_size, linewidth)

    class FenixFancy(_Base):
        """A fancy arrow. Only works with a quadratic Bézier curve."""

        def __init__(
            self,
            head_length=0.4,
            head_width=0.4,
            tail_width=0.4,
            tpercent=0.15,
            fancyness=0.3,
        ):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.4
                Width of the arrow head.

            tail_width : float, default: 0.4
                Width of the arrow tail.

            tpercent : float, default: 0.15
                thickness of arrow.

            fancyness : float, default: 0.3
                fancyness of arrow
            """
            self.head_length, self.head_width, self.tail_width, self.tpercent, self.fancyness = (
                head_length,
                head_width,
                tail_width,
                tpercent,
                fancyness
            )
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length = self.head_length * mutation_size
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            # path for head
            in_f = inside_circle(x2, y2, head_length)
            try:
                path_out, path_in = split_bezier_intersecting_with_closedpath(
                    arrow_path, in_f
                )
            except NonIntersectingPathException:
                # if this happens, make a straight line of the head_length
                # long.
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
                arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
                path_head = arrow_path
            else:
                path_head = path_in

            # path for head
            in_f = inside_circle(x2, y2, head_length * 0.8)
            path_out, path_in = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f
            )
            path_tail = path_out

            # head
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(path_head, head_width / 2.0, wm=0.6)

            # tail
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(
                path_tail, head_width / 2.0, wm=0.6
            )

            # make_wedged_bezier2(path_tail,
            #                    tail_width * .5,
            #                    w1=1., wm=0.6, w2=0.3)

            # path for head
            in_f = inside_circle(x0, y0, tail_width * 0.3)
            path_in, path_out = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f
            )
            tail_start = path_in[-1]

            head_right, head_left = head_r, head_l

            head_right = np.array(head_right)
            head_left = np.array(head_left)
            tail_right = np.array(tail_right)
            tail_left = np.array(tail_left)

            tpercent = self.tpercent
            fancyness = self.fancyness
            tip = head_right[2]
            dir = 0.5 * (head_right[0] + head_left[0]) - head_right[2]
            hr = head_right[0] + fancyness * dir
            hl = head_left[0] + fancyness * dir
            ml = 0.5 * (head_right[0] + head_left[0]) + tpercent * (
                head_left[0] - head_right[0]
            )
            mr = 0.5 * (head_right[0] + head_left[0]) - tpercent * (
                head_left[0] - head_right[0]
            )
            tl = 0.5 * (tail_left[0] + tail_right[0]) + tpercent * (
                tail_left[0] - tail_right[0]
            )
            tr = 0.5 * (tail_left[0] + tail_right[0]) - tpercent * (
                tail_left[0] - tail_right[0]
            )

            patch_path = [
                (Path.MOVETO, tip),
                (Path.LINETO, hr),
                (Path.LINETO, mr),
                (Path.LINETO, tr),
                (Path.LINETO, tl),
                (Path.LINETO, ml),
                (Path.LINETO, hl),
                (Path.LINETO, tip),
                (Path.CLOSEPOLY, tip),
            ]

            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    _style_list = {"fenix": FenixFancy}
