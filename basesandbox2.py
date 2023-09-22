import functools
import random
import time
import numpy
from numba import prange, njit, cuda


BASE_COLOR = (15, 20, 25)

subs = {
    'gas': {
        "diffuse": True,
        "bulk": False,
        "liquid": False,
    },
    'bulk': {
        "diffuse": False,
        "bulk": True,
        "liquid": False,
    },
    'liquid': {
        "diffuse": False,
        "bulk": False,
        "liquid": True,
    },
}

statuses = {
    'temperature': int,
    'temperature2': int,
    'temperature3': int,
    'temperature4': int,
    'temperature5': int,
    'temperature6': int,
    'temperature7': int,
    'temperature8': int,
    'temperature9': int,
    'temperature10': int,
}

pixs = (
    {
        'density': 1.42897,  # kg/m3
        'color': BASE_COLOR,
        'name': 'Air',
        "subs": 0,
    },
    {
        'density': 1550.0,  # kg/m3
        'color': (200, 150, 10),
        'name': 'Sand',
        "subs": 1,
    },
    {
        'density': 1000.0,  # kg/m3
        'color': (35, 40, 210),
        'name': 'Water',
        "subs": 2,
    },
    {
        'density': 1.0,  # kg/m3
        'color': (45, 50, 55),
        'name': 'Helium',
        "subs": 0,
    },
)

COLORS = tuple(info['color'] for info in pixs)
NAMES = tuple(info['name'] for info in pixs)
SUBS = tuple(info['subs'] for info in pixs)
DENSITY = tuple(info['density'] for info in pixs)
SUBS_CHARS = tuple(tuple(val for val in sub.values()) for sub in subs.values())

num_statuses = len(statuses)
default_statuses = (*[0] * num_statuses,)

device = cuda.get_current_device()


class Utils:
    @staticmethod
    def speedtest(func):
        def wrap(*args, **kwargs):
            start = time.time()

            try:
                res = func(*args, **kwargs)
            except Exception as e:
                print(e)
                print(e.args)
                raise e

            eps_time = time.time() - start

            if eps_time == 0:
                print(f"{eps_time:.8f}")
            else:
                print(f"{eps_time:.8f} – {1 / eps_time}q/s")
            return res

        return wrap

    @staticmethod
    def circle_center(func):
        @functools.wraps(func)
        def wrap(r: int, x0: int = 0, y0: int = 0):
            if not (x0 or y0):
                return func(r)

            res: numpy.ndarray = func(r)
            return res + numpy.array([(x0, y0)] * res.shape[0])

        return wrap

    @staticmethod
    @circle_center
    @functools.cache
    @njit
    def get_pixels_on_circle(r: int):
        pixels_on_circle = []

        for x in range(1 - r, r):
            for y in range(1 - r, r):
                if x ** 2 + y ** 2 <= r ** 2:
                    pixels_on_circle.append((x, y))
        return numpy.array(pixels_on_circle)

    @staticmethod
    @njit(parallel=True)
    def place_pixels(width: int, height: int, pmatrix: numpy.ndarray,
                     selected_pix: int, points, x0: int = 0, y0: int = 0) -> None:
        for ind in prange(len(points)):
            if (0 < x0 + points[ind][0] < width) and (0 < y0 + points[ind][1] < height):
                pox_mx, pos_my = points[ind]
                pmatrix[x0 + pox_mx, y0 + pos_my] = (selected_pix, *default_statuses)

    @staticmethod
    def place_pixels_many(width: int, height: int, pmatrix: numpy.ndarray,
                          selected_pix: int, points, x0: int = 0, y0: int = 0) -> None:
        for pnts in points:
            place_pixels(width, height, pmatrix, selected_pix, pnts, x0, y0)

    @staticmethod
    def get_line_points(x0: int, y0: int, x1: int, y1: int):
        dx, dy = abs(x1 - x0), abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        current_x, current_y = x0, y0

        while current_x != x1 or current_y != y1:
            yield current_x, current_y

            err2 = 2 * err

            if err2 > -dy:
                err -= dy
                current_x += sx

            if err2 < dx:
                err += dx
                current_y += sy

    @staticmethod
    @functools.cache
    @njit
    def get_color(idx: tuple[int]) -> tuple[int, int, int]:
        return COLORS[idx[0]]


class Matrix:
    def __init__(self, size: tuple[int, int], fill_pix: int = 0):
        self.size = size
        self.selected_pix = 0

        self.pmatrix = numpy.array([[(fill_pix, *default_statuses)] * self.size[1]] * self.size[0])

    def select_pixel(self, pix_id: int) -> None:
        self.selected_pix = pix_id

    @staticmethod
    @njit(parallel=True)
    def get_color_array(pmatrix: numpy.ndarray, size: tuple[int, int]):
        colors_array = numpy.array([[(0, 0, 0)] * size[1]] * size[0])

        for xind in prange(len(pmatrix)):
            for yind in prange(len(pmatrix[xind])):
                colors_array[xind, yind] = COLORS[pmatrix[xind, yind, 0]]
        return colors_array

    def __getitem__(self, key):
        if len(key) == 1:
            return self.pmatrix[key[0]]
        else:
            return self.pmatrix[key[0], key[1]]

    def __setitem__(self, key, value):
        if len(key) == 1:
            self.pmatrix[key[0]] = value
        else:
            self.pmatrix[key[0], key[1]] = value

    @staticmethod
    @Utils.speedtest
    @njit(parallel=True, fastmath=True)
    def iter(pmatrix: numpy.ndarray):
        for x in prange(pmatrix.shape[0]):
            for y in prange(pmatrix.shape[1]):
                pixtypes = SUBS_CHARS[SUBS[pmatrix[x, y, 0]]]
                pixdensity = DENSITY[pmatrix[x, y, 0]]

                if pixtypes[0]:
                    if x == 0:
                        continue

                    if 0 < y + 1 < pmatrix.shape[1]:
                        y_c = y + 1
                        x_c = x
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            if random.random() < 1 / 3:
                                temp = pmatrix[x_c, y_c].copy()
                                pmatrix[x_c, y_c] = pmatrix[x, y]
                                pmatrix[x, y] = temp
                    if 0 < x - 1 < pmatrix.shape[0]:
                        y_c = y
                        x_c = x - 1
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            if random.random() < 1 / 5:
                                temp = pmatrix[x_c, y_c].copy()
                                pmatrix[x_c, y_c] = pmatrix[x, y]
                                pmatrix[x, y] = temp
                    if 0 < x + 1 < pmatrix.shape[0]:
                        y_c = y
                        x_c = x + 1
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            if random.random() < 1 / 3:
                                temp = pmatrix[x_c, y_c].copy()
                                pmatrix[x_c, y_c] = pmatrix[x, y]
                                pmatrix[x, y] = temp
                    if 0 < y + 1 < pmatrix.shape[1] and 0 < x + 1 < pmatrix.shape[0]:
                        y_c = y + 1
                        x_c = x + 1
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            if random.random() < 1 / 3:
                                temp = pmatrix[x_c, y_c].copy()
                                pmatrix[x_c, y_c] = pmatrix[x, y]
                                pmatrix[x, y] = temp
                    if 0 < y + 1 < pmatrix.shape[1] and 0 < x - 1 < pmatrix.shape[0]:
                        y_c = y + 1
                        x_c = x - 1
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            if random.random() < 1 / 5.65:
                                temp = pmatrix[x_c, y_c].copy()
                                pmatrix[x_c, y_c] = pmatrix[x, y]
                                pmatrix[x, y] = temp
                elif pixtypes[1]:
                    if y == pmatrix.shape[1]:
                        continue

                    if 0 < y + 1 < pmatrix.shape[1]:
                        y_c = y + 1
                        x_c = x
                        if DENSITY[pmatrix[x_c, y_c, 0]] < pixdensity:
                            temp = pmatrix[x_c, y_c].copy()
                            pmatrix[x_c, y_c] = pmatrix[x, y]
                            pmatrix[x, y] = temp
                        else:
                            lx_c = x - 1
                            rx_c = x + 1
                            y_c = x - 1

                            leftdens = DENSITY[pmatrix[lx_c, y - 1, 0]]
                            rightdens = DENSITY[pmatrix[rx_c, y - 1, 0]]

                            if leftdens < pixdensity or rightdens < pixdensity:
                                if leftdens < pixdensity and rightdens < pixdensity:
                                    if random.random() <= 0.5:
                                        temp = pmatrix[rx_c, y_c].copy()
                                        pmatrix[rx_c, y_c] = pmatrix[x, y]
                                        pmatrix[x, y] = temp
                                    else:
                                        temp = pmatrix[lx_c, y_c].copy()
                                        pmatrix[lx_c, y_c] = pmatrix[x, y]
                                        pmatrix[x, y] = temp


place_pixels = Utils.place_pixels
