import json
import time

import numpy
import pygame
from numba import prange, njit

from pixmove import check_borders, move_cond

lang = "ru"
language = json.load(open("langs.json", 'r', encoding="UTF-8"))[lang]

BASE_COLOR = (15, 20, 25)

CtoK = lambda temp: temp + 273.15
DEFAULT_TEMPERATURE = CtoK(25)

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
    'static': {
        "diffuse": False,
        "bulk": False,
        "liquid": False,
    },
}

pixs = (
    {
        # 'density': 0.0,  # kg/m3
        'density': 1.42897,  # kg/m3
        'color': BASE_COLOR,
        'name': 'air',
        "subs": 0,
        "heat_capacity": 1.005,
    },
    {
        'density': 1550.0,  # kg/m3
        'color': (230, 185, 20),
        'name': 'snd',
        "subs": 1,
        "heat_capacity": 0.84,
    },
    {
        'density': 1000.0,  # kg/m3
        'color': (35, 40, 210),
        'name': 'wtr',
        "subs": 2,
        "heat_capacity": 4.184,
    },
    {
        'density': 1100.0,  # kg/m3
        'color': (255, 40, 210),
        'name': 'wtr',
        "subs": 2,
        "heat_capacity": 4.184,
    },
    {
        'density': 0.173,  # kg/m3
        'color': (45, 50, 55),
        'name': 'hel',
        "subs": 0,
        "heat_capacity": 5.19,
    },
    {
        'density': 3.209,  # kg/m3
        'color': (180, 245, 40),
        'name': 'chlr',
        "subs": 0,
        "heat_capacity": 0.52,
    },
    {
        'density': 0.08987,  # kg/m3
        'color': (240, 70, 145),
        'name': 'hdrg',
        "subs": 0,
        "heat_capacity": 14.27,
    },
    {
        'density': 2600.0,  # kg/m3
        'color': (225, 50, 20),
        'name': 'lava',
        "subs": 2,
        "temperature": CtoK(1100),
        "heat_capacity": 1.45,
    },
    {
        'density': 808.0,  # kg/m3
        'color': (210, 240, 250),
        'name': 'lqntg',
        "subs": 2,
        "temperature": CtoK(-205),
        "heat_capacity": 1.04,
    },
    {
        'density': 1.695,  # kg/m3
        'color': (255, 245, 130),
        'name': 'flr',
        "subs": 0,
        "heat_capacity": 0.824,
    },
    {
        'density': 999999999.9,  # kg/m3
        'color': (215, 90, 75),
        'name': 'brck',
        "subs": 3,
        "heat_capacity": 0.84,
    },
    {
        # 'density': 917.0,  # kg/m3
        'density': 999999999.9,  # kg/m3
        'color': (125, 140, 235),
        'name': 'ice',
        "subs": 3,
        "temperature": CtoK(-15),
        "heat_capacity": 2.093,
    },
)

COLORS = tuple(info['color'] for info in pixs)
NAMES = tuple(language['subs'][info['name']] for info in pixs)
SUBS = tuple(info['subs'] for info in pixs)
DENSITY = tuple(info['density'] for info in pixs)
HEAT_CAPACITY = tuple(info['heat_capacity'] for info in pixs)
SUBS_CHARS = tuple(tuple(val for val in sub.values()) for sub in subs.values())
PIXS_TEMPERATURES = tuple(pix.get("temperature", DEFAULT_TEMPERATURE) for pix in pixs)

points = (
    (5, (0, 0, 255)),
    (285, (15, 0, 0)),
    (400, (160, 0, 0)),
    (685, (255, 145, 0)),
    (1650, (255, 255, 255))
)


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
        def wrap(r: int, x0: int = 0, y0: int = 0):
            if x0 == 0 and y0 == 0:
                return func(r)

            res = func(r)
            return res + numpy.array([(x0, y0)] * res.shape[0])

        return wrap

    @staticmethod
    @circle_center
    @njit(fastmath=True, cache=True)
    def get_pixels_on_circle(r: int):
        pixels_on_circle = numpy.array([(0, 0)] * int(3.1415 * r ** 2))

        for x_c in prange(0, r * 2):
            for y_c in prange(0, r * 2):
                if (x_c - r) ** 2 + (y_c - r) ** 2 <= r ** 2:
                    pixels_on_circle[x_c * r + y_c] = ((x_c - r), (y_c - r))

        return pixels_on_circle

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def drawline(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, bool_array: numpy.ndarray, x0: int, y0: int,
                 x1: int, y1: int, brushsize: int, selected_pix: int,
                 brush_mode: int, heat_quan: int, mtrx_w: int, mtrx_h: int):
        dx: int = abs(x1 - x0)
        dy: int = abs(y1 - y0)
        sx: int = 1 if x0 < x1 else -1
        sy: int = 1 if y0 < y1 else -1
        err: int = dx - dy

        while True:
            for i in prange(-brushsize, brushsize + 1):
                for j in prange(-brushsize, brushsize + 1):
                    x: int = x0 + i
                    y: int = y0 + j

                    if 0 <= x < mtrx_w and 0 <= y < mtrx_h:
                        if brush_mode == 0:
                            pmatrix[x, y] = selected_pix
                            bool_array[x, y] = True
                            temp_pmatrix[x, y] = PIXS_TEMPERATURES[selected_pix]
                        elif brush_mode == 1:
                            temp_pmatrix[x, y] = max(temp_pmatrix[x, y] + heat_quan, 0)

            if x0 == x1 and y0 == y1:
                break

            e2: int = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx

            if x0 == x1 and y0 == y1:
                for i in prange(-brushsize, brushsize + 1):
                    for j in prange(-brushsize, brushsize + 1):
                        x: int = x0 + i
                        y: int = y0 + j

                        if 0 <= x < mtrx_w and 0 <= y < mtrx_h:
                            if brush_mode == 0:
                                pmatrix[x, y] = selected_pix
                                bool_array[x, y] = True
                                temp_pmatrix[x, y] = PIXS_TEMPERATURES[selected_pix]
                            elif brush_mode == 1:
                                temp_pmatrix[x, y] = max(temp_pmatrix[x, y] + heat_quan, 0)
                break

            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    @njit(parallel=True, nogil=True)
    def place_pixels(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, selected_pix: int, brush_mode: int,
                     heat_quan: int, points, x0: int = 0, y0: int = 0) -> numpy.ndarray:
        for ind in prange(len(points)):
            if (0 <= x0 + points[ind][0] < pmatrix.shape[0]) and (0 <= y0 + points[ind][1] < pmatrix.shape[1]):
                pox_mx, pos_my = points[ind]

                if brush_mode == 0:
                    pmatrix[x0 + pox_mx, y0 + pos_my] = selected_pix
                    temp_pmatrix[x0 + pox_mx, y0 + pos_my] = PIXS_TEMPERATURES[selected_pix]
                elif brush_mode == 1:
                    temp = max(temp_pmatrix[x0 + pox_mx, y0 + pos_my] + heat_quan, 0)
                    temp_pmatrix[x0 + pox_mx, y0 + pos_my] = temp

    @staticmethod
    # @njit(parallel=True, nogil=True)
    def place_pixels_many(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, color_array_bool: numpy.ndarray,
                          selected_pix: int, brush_mode: int, heat_quan: int, points,
                          x0: int = 0, y0: int = 0) -> numpy.ndarray:
        for pnts_ind in prange(len(points)):
            cur_points = points[pnts_ind]

            place_pixels(pmatrix, temp_pmatrix, selected_pix, brush_mode, heat_quan, cur_points, x0, y0)

            for ind in prange(len(cur_points)):
                if (0 <= x0 + cur_points[ind][0] < pmatrix.shape[0]) and \
                        (0 <= y0 + cur_points[ind][1] < pmatrix.shape[1]):
                    pox_mx, pos_my = cur_points[ind]
                    color_array_bool[x0 + pox_mx, y0 + pos_my] = True
        return color_array_bool

    @staticmethod
    @njit(fastmath=True, cache=True)
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
    @njit(fastmath=True, cache=True)
    def temperature_to_color(temp: int) -> tuple[int, int, int]:
        if temp <= points[1][0]:
            blue = points[0][1][2] + (temp - points[0][0]) / (points[1][0] - points[0][0]) * (
                    points[1][1][2] - points[0][1][2])
            green = points[0][1][1] + (temp - points[0][0]) / (points[1][0] - points[0][0]) * (
                    points[1][1][1] - points[0][1][1])
            red = points[0][1][0] + (temp - points[0][0]) / (points[1][0] - points[0][0]) * (
                    points[1][1][0] - points[0][1][0])

            return int(red), int(green), int(blue)
        elif temp <= points[2][0]:
            blue = points[1][1][2] + (temp - points[1][0]) / (points[2][0] - points[1][0]) * (
                    points[2][1][2] - points[1][1][2])
            green = points[1][1][1] + (temp - points[1][0]) / (points[2][0] - points[1][0]) * (
                    points[2][1][1] - points[1][1][1])
            red = points[1][1][0] + (temp - points[1][0]) / (points[2][0] - points[1][0]) * (
                    points[2][1][0] - points[1][1][0])

            return int(red), int(green), int(blue)
        elif temp <= points[3][0]:
            blue = points[2][1][2] + (temp - points[2][0]) / (points[3][0] - points[2][0]) * (
                    points[3][1][2] - points[2][1][2])
            green = points[2][1][1] + (temp - points[2][0]) / (points[3][0] - points[2][0]) * (
                    points[3][1][1] - points[2][1][1])
            red = points[2][1][0] + (temp - points[2][0]) / (points[3][0] - points[2][0]) * (
                    points[3][1][0] - points[2][1][0])

            return int(red), int(green), int(blue)
        elif temp <= points[4][0]:
            blue = points[3][1][2] + (temp - points[3][0]) / (points[4][0] - points[3][0]) * (
                    points[4][1][2] - points[3][1][2])
            green = points[3][1][1] + (temp - points[3][0]) / (points[4][0] - points[3][0]) * (
                    points[4][1][1] - points[3][1][1])
            red = points[3][1][0] + (temp - points[3][0]) / (points[4][0] - points[3][0]) * (
                    points[4][1][0] - points[3][1][0])

            return int(red), int(green), int(blue)
        return 255, 255, 255


class Matrix:
    def __init__(self, size: tuple[int, int], fill_pix: int = 0):
        self.size = size
        self.selected_pix = 0
        self.display_mode = 1
        self.fill_pix = fill_pix
        self.brush_mode = 1
        self.pause = False

        self.pmatrix = numpy.array(numpy.full(size, 0, dtype=int))
        self.temp_pmatrix = numpy.array([[DEFAULT_TEMPERATURE] * self.size[1]] * self.size[0], dtype=float)

        self.colors_array_bool = numpy.array(numpy.full(size, True))
        self.colors_array, self.colors_array_bool = self.get_color_array(numpy.array([[(0, 0, 0)] * size[1]] * size[0]),
                                                                         self.colors_array_bool, self.pmatrix,
                                                                         self.temp_pmatrix, self.display_mode)

        self.surface = pygame.Surface(size)
        psx3d = pygame.surfarray.pixels3d(self.surface)

        psx3d[:] = self.colors_array
        del psx3d

    def reset_field(self):
        self.pmatrix = numpy.array(numpy.full(self.size, 0, dtype=int))
        self.temp_pmatrix = numpy.array([[DEFAULT_TEMPERATURE] * self.size[1]] * self.size[0], dtype=float)

        self.colors_array_bool = numpy.array(numpy.full(self.size, True))
        self.colors_array, self.colors_array_bool = self.get_color_array(
            numpy.array([[(0, 0, 0)] * self.size[1]] * self.size[0]), self.colors_array_bool, self.pmatrix,
            self.temp_pmatrix, self.display_mode)

        self.surface = pygame.Surface(self.size)
        psx3d = pygame.surfarray.pixels3d(self.surface)

        psx3d[:] = self.colors_array
        del psx3d

    def set_display_mode(self) -> None:
        self.display_mode = 1 if self.display_mode == 0 else 0
        self.colors_array_bool = numpy.full(self.colors_array_bool.shape, True)

    def set_pause(self) -> None:
        self.pause = not self.pause

    def set_brush_mode(self) -> None:
        self.brush_mode = 1 if self.brush_mode == 0 else 0

    def select_pixel(self, pix_id: int) -> None:
        self.selected_pix = pix_id

    @staticmethod
    @njit(parallel=True, nogil=True, fastmath=True)
    def get_color_array(colors_array: numpy.ndarray, colors_array_bools: numpy.ndarray, pmatrix: numpy.ndarray,
                        temp_pmatrix: numpy.ndarray, mode: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        shape = pmatrix.shape

        for xind in prange(shape[0]):
            for yind in prange(shape[1]):
                if colors_array_bools[xind, yind]:
                    if mode == 0:
                        colors_array[xind, yind] = COLORS[pmatrix[xind, yind]]
                    elif mode == 1:
                        colors_array[xind, yind] = temperature_to_color(temp_pmatrix[xind, yind])
        return colors_array, numpy.full(colors_array_bools.shape, False)

    def __getitem__(self, key):
        return self.pmatrix[*key]

    def __setitem__(self, key, value):
        self.pmatrix[*key] = value

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def changing_temp(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray,
                      pmatrix_bools: numpy.ndarray, x0: int, y0: int, x1: int, y1: int, heat_coef: float) -> None:
        if not (0 <= x0 + x1 < temp_pmatrix.shape[0] and 0 <= y0 + y1 < temp_pmatrix.shape[1]):
            return

        n0 = temp_pmatrix[x0, y0]
        n1 = temp_pmatrix[x0 + x1, y0 + y1]

        if n0 == n1:
            return

        dens1 = DENSITY[pmatrix[x0, y0]]
        dens2 = DENSITY[pmatrix[x0 + x1, y0 + y1]]

        heat_cap1 = HEAT_CAPACITY[pmatrix[x0, y0]]
        heat_cap2 = HEAT_CAPACITY[pmatrix[x0 + x1, y0 + y1]]

        # n0k = HEAT_CAPACITY[pmatrix[x0, y0, 0]]
        # n1k = HEAT_CAPACITY[pmatrix[x0 + x1, y0 + y1, 0]]

        diffChars = (heat_cap1 / heat_cap2) ** 1.15 / dens1 * dens2

        if diffChars > 1:
            diffChars = 1 / diffChars

        diff = abs(n0 - (n0 + n1) / 2) * heat_coef * diffChars

        if n0 > n1:
            temp_pmatrix[x0, y0] -= diff
            temp_pmatrix[x0 + x1, y0 + y1] += diff
        elif n0 < n1:
            temp_pmatrix[x0, y0] += diff
            temp_pmatrix[x0 + x1, y0 + y1] -= diff

        pmatrix_bools[x0, y0] = True
        pmatrix_bools[x0 + x1, y0 + y1] = True

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def temp_iter(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, pmatrix_bools: numpy.ndarray,
                  heat_coef: float) -> numpy.ndarray:
        shape = temp_pmatrix.shape
        # points = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        points = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, 1)]
        ln = len(points)
        changing_temp = global_changing_temp

        for x in prange(shape[0]):
            for y in range(shape[1]):
                for pind in range(ln):
                    changing_temp(pmatrix, temp_pmatrix, pmatrix_bools, x, y, points[pind][0], points[pind][1],
                                  heat_coef)

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def iter(pmatrix_bools: numpy.ndarray, pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray) -> numpy.ndarray:
        shape = pmatrix.shape

        for x in prange(shape[0]):
            for y in prange(shape[1]):
                pixtypes = SUBS_CHARS[SUBS[pmatrix[x, y]]]
                pixdensity = DENSITY[pmatrix[x, y]]

                if pixtypes[0]:
                    move_cond(x, y, x, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY, 1 / 3)
                    move_cond(x, y, x - 1, y, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY, 1 / 5)
                    move_cond(x, y, x + 1, y, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY, 1 / 3)
                    move_cond(x, y, x + 1, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY,
                              1 / 3)
                    move_cond(x, y, x - 1, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY,
                              1 / 5.65)
                elif pixtypes[1]:
                    down_cond = 0

                    if check_borders(x, y + 1, shape):
                        if DENSITY[pmatrix[x, y + 1]] < pixdensity:
                            down_cond += 1

                    move_cond(x, y, x - 1, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY,
                              1 / 5 / (down_cond + 1))
                    move_cond(x, y, x + 1, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix, DENSITY,
                              1 / 3.8 / (down_cond + 1))

                    if down_cond:
                        pmatrix[x, y], pmatrix[x, y + 1] = pmatrix[x, y + 1], pmatrix[x, y]
                        pmatrix_bools[x, y] = True
                        pmatrix_bools[x, y + 1] = True
                elif pixtypes[2]:
                    can_do = move_cond(x, y, x, y + 1, shape, pixdensity, pmatrix, pmatrix_bools, temp_pmatrix,
                                       DENSITY, 1 / 1.15, ret_posdens_bool=True)

                    if can_do:
                        if check_borders(x + 1, y, shape) and check_borders(x - 1, y, shape):
                            left_density = DENSITY[pmatrix[x - 1, y]]
                            right_density = DENSITY[pmatrix[x + 1, y]]

                            if left_density >= pixdensity and right_density >= pixdensity:
                                continue

                        left = 0
                        right = 0

                        while x + left - 1 >= 0:
                            den = DENSITY[pmatrix[x + left - 1, y + 1]]

                            if den < pixdensity:
                                left -= 1
                                break
                            if den == pixdensity:
                                left -= 1
                            else:
                                break

                        while x + right + 1 < shape[0]:
                            den = DENSITY[pmatrix[x + right + 1, y + 1]]

                            if den < pixdensity:
                                right += 1
                                break
                            if den == pixdensity:
                                right += 1
                            else:
                                break

                        right_max = x + right == shape[0]
                        left_max = x + left == 0

                        if (right_max and left_max) or (left == 0 and right == 0):
                            continue
                        elif right_max and right != 0:
                            x_c = x + left
                        elif left_max and left != 0:
                            x_c = x + right
                        else:
                            x_c = x + (right if abs(right) < abs(left) else left)

                        pmatrix[x, y], pmatrix[x_c, y] = pmatrix[x_c, y], pmatrix[x, y]
                        pmatrix_bools[x, y] = True
                        pmatrix_bools[x_c, y] = True

        return pmatrix_bools


place_pixels = Utils.place_pixels
temperature_to_color = Utils.temperature_to_color
global_changing_temp = Matrix.changing_temp
