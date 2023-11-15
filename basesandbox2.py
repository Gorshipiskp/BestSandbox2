import json
import time
from random import random

import numpy
import pygame
from numba import prange, njit

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
        'temp_influence': True,
        'static': False
    },
    'bulk': {
        "diffuse": False,
        "bulk": True,
        "liquid": False,
        'temp_influence': True,
        'static': False
    },
    'liquid': {
        "diffuse": False,
        "bulk": False,
        "liquid": True,
        'temp_influence': True,
        'static': False
    },
    'static': {
        "diffuse": False,
        "bulk": False,
        "liquid": False,
        'temp_influence': True,
        'static': True
    },
    'full-static': {
        "diffuse": False,
        "bulk": False,
        "liquid": False,
        'temp_influence': False,
        'static': True
    },
}

pixs = (
    {  # 0
        # 'density': 0.0,  # kg/m3
        'density': 10,  # kg/m3
        'color': BASE_COLOR,
        'name': 'empty',
        "temperature": CtoK(15),
        "subs": 0,
        "heat_capacity": 0,

    },
    {  # 1
        'density': 1550.0,  # kg/m3
        'color': (230, 185, 20),
        'name': 'snd',
        "subs": 1,
        "heat_capacity": 0.84,
    },
    {  # 2
        'density': 1000.0,  # kg/m3
        'color': (35, 40, 210),
        'name': 'wtr',
        "subs": 2,
        "heat_capacity": 4.184,
        "temp_transformations": (CtoK(0), 10, CtoK(100), 11, 2)
    },
    {  # 3
        'density': 0.173,  # kg/m3
        'color': (45, 50, 55),
        'name': 'hel',
        "subs": 0,
        "heat_capacity": 5.19,
    },
    {  # 4
        'density': 3.209,  # kg/m3
        'color': (180, 245, 40),
        'name': 'chlr',
        "subs": 0,
        "heat_capacity": 0.52,
    },
    {  # 5
        'density': 0.08987,  # kg/m3
        'color': (240, 70, 145),
        'name': 'hdrg',
        "subs": 0,
        "heat_capacity": 14.27,
    },
    {  # 6
        'density': 2600.0,  # kg/m3
        'color': (225, 50, 20),
        'name': 'lava',
        "subs": 2,
        "temperature": CtoK(1100),
        "heat_capacity": 1.45,
    },
    {  # 7
        'density': 808.0,  # kg/m3
        'color': (195, 225, 235),
        'name': 'lqntg',
        "subs": 2,
        "temperature": CtoK(-205),
        "temp_transformations": (CtoK(-215), 14, CtoK(-200), 13, 7),
        "heat_capacity": 20.04,
        # "heat_capacity": 1.04,
    },
    {  # 8
        'density': 1.695,  # kg/m3
        'color': (255, 245, 130),
        'name': 'flr',
        "subs": 0,
        "heat_capacity": 0.824,
    },
    {  # 9
        'density': 999999.9,  # kg/m3
        'color': (215, 90, 75),
        'name': 'brck',
        "subs": 4,
        "heat_capacity": 0.84,
    },
    {  # 10
        'density': 917.0,  # kg/m3
        'color': (125, 140, 235),
        'name': 'ice',
        "subs": 3,
        "temperature": CtoK(-15),
        "heat_capacity": 2.093,
        "temp_transformations": (CtoK(0), 10, CtoK(100), 11, 2)
    },
    {  # 11
        'density': 1.2,  # kg/m3
        'color': (155, 165, 240),
        'name': 'steam',
        "temperature": CtoK(120),
        "subs": 0,
        "heat_capacity": 1.996,
        "temp_transformations": (CtoK(0), 10, CtoK(100), 11, 2)
    },
    {  # 12
        'density': 1200000.0,  # kg/m3
        'color': (165, 165, 165),
        'name': 'brck_thermo',
        "subs": 3,
        "heat_capacity": 10000,
    },
    {  # 13
        'density': 1.2506,  # kg/m3
        'color': (105, 105, 105),
        'name': 'nitrogen',
        "subs": 0,
        "heat_capacity": 1.042,
        "temp_transformations": (CtoK(-215), 14, CtoK(-200), 13, 7),
    },
    {  # 14
        'density': 1024,  # kg/m3
        'color': (155, 155, 155),
        'name': 'solid_nitrogen',
        "subs": 1,
        "heat_capacity": 1.042,
        "temp_transformations": (CtoK(-215), 14, CtoK(-200), 13, 7),
    },
    {  # 15
        'density': 1200.0,  # kg/m3
        'color': (255, 45, 45),
        'name': 'heater',
        "subs": 3,
        "heat_capacity": 100,
    },
    {  # 16
        'density': 1200.0,  # kg/m3
        'color': (45, 45, 255),
        'name': 'cooler',
        "subs": 3,
        "heat_capacity": 100,
    },
    {  # 16
        'density': 99999999999999999.0,  # kg/m3
        'color': (0, 0, 0),
        'name': 'filler',
        "subs": 4,
        "heat_capacity": 1,
    },
)

# textures = process_textures()

TEMPERATURE_TRANSFORMATIONS = numpy.array(tuple(info.get('temp_transformations', (-666, 0, 0, 0, 0)) for info in pixs))
COLORS = numpy.array(tuple(info['color'] for info in pixs))
NAMES = numpy.array(tuple(language['subs'][info['name']] for info in pixs))
DENSITY = numpy.array(tuple(info['density'] for info in pixs))
HEAT_CAPACITY = numpy.array(tuple(info['heat_capacity'] for info in pixs))
SUBS = numpy.array(tuple(info['subs'] for info in pixs))
SUBS_CHARS = numpy.array(tuple(numpy.array(tuple(val for val in sub.values())) for sub in subs.values()))
SUBS_CHARS_PIXS = numpy.array(tuple(SUBS_CHARS[sub] for sub in SUBS))
# PIXS_TEXTURES = numba.typed.List([textures.get(info['name'], numpy.array([[[-1, -1, -1]]])) for info in pixs])
PIXS_TEMPERATURES = numpy.array(tuple(pix.get("temperature", DEFAULT_TEMPERATURE) for pix in pixs))

points = (
    (0, (0, 0, 255)),
    (285, (15, 0, 0)),
    (385, (160, 0, 0)),
    (685, (255, 145, 0)),
    (1650, (255, 255, 255))
)


@njit(nogil=True, cache=True, fastmath=True)
def can_move(x: int, y: int, shape_x: int, shape_y: int, typepix1: int, typepix2: int):
    return check_borders(x, y, shape_x, shape_y) and not SUBS_CHARS_PIXS[typepix2, 4] and DENSITY[typepix1] > DENSITY[typepix2]


@njit(fastmath=True, nogil=True)
def move_cond(x0: int, y0: int, x1: int, y1: int, shape_x: int, shape_y: int, pixdensity: float, pmatrix: numpy.ndarray,
              temp_pmatrix: numpy.ndarray, possibility: float = 1.0):
    if check_borders(x1, y1, shape_x, shape_y) and not SUBS_CHARS_PIXS[pmatrix[x1, y1], 4] and \
            DENSITY[pmatrix[x1, y1]] < pixdensity:
        if random() <= possibility:
            pmatrix[x0, y0], pmatrix[x1, y1] = pmatrix[x1, y1], pmatrix[x0, y0]
            temp_pmatrix[x1, y1], temp_pmatrix[x0, y0] = temp_pmatrix[x0, y0], temp_pmatrix[x1, y1]


@njit(fastmath=True, cache=True, nogil=True)
def check_borders(x: int, y: int, shape_x: int, shape_y: int) -> bool:
    return 0 < x < shape_x - 1 and 0 < y < shape_y - 1


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
    @njit(parallel=True, fastmath=True, nogil=True)
    def drawline(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, x0: int, y0: int,
                 x1: int, y1: int, brushsize: int, selected_pix: int,
                 brush_mode: int, heat_quan: int, mtrx_w: int, mtrx_h: int):
        dx: int = numpy.abs(x1 - x0)
        dy: int = numpy.abs(y1 - y0)
        sx: int = 1 if x0 < x1 else -1
        sy: int = 1 if y0 < y1 else -1
        err: int = dx - dy

        while True:
            for i in prange(-brushsize, brushsize + 1):
                for j in prange(-brushsize, brushsize + 1):
                    x: int = x0 + i
                    y: int = y0 + j

                    if check_borders(x, y, mtrx_w, mtrx_h):
                        if brush_mode == 0:
                            pmatrix[x, y] = selected_pix
                            temp_pmatrix[x, y] = PIXS_TEMPERATURES[selected_pix]
                        elif brush_mode == 1:
                            if pmatrix[x, y]:
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

                        if check_borders(x, y, mtrx_w, mtrx_h):
                            if brush_mode == 0:
                                pmatrix[x, y] = selected_pix
                                temp_pmatrix[x, y] = PIXS_TEMPERATURES[selected_pix]
                            elif brush_mode == 1:
                                if pmatrix[x, y]:
                                    temp_pmatrix[x, y] = max(temp_pmatrix[x, y] + heat_quan, 0)
                break

            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
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
        self.display_mode = 0
        self.fill_pix = fill_pix
        self.brush_mode = 0
        self.pause = False

        self.pmatrix = numpy.array(numpy.full(size, 0, dtype=int))
        self.temp_pmatrix = numpy.array(numpy.full(self.size, PIXS_TEMPERATURES[fill_pix], dtype=float))

        self.colors_array = self.get_color_array(numpy.array([[(0, 0, 0)] * size[1]] * size[0]), self.pmatrix,
                                                 self.temp_pmatrix, self.display_mode)

        self.surface = pygame.Surface(size)
        psx3d = pygame.surfarray.pixels3d(self.surface)

        psx3d[:] = self.colors_array
        del psx3d

    def reset_field(self):
        self.pmatrix = numpy.array(numpy.full(self.size, 0, dtype=int))
        self.temp_pmatrix.fill(PIXS_TEMPERATURES[self.fill_pix])

        self.colors_array = self.get_color_array(
            numpy.array([[(0, 0, 0)] * self.size[1]] * self.size[0]), self.pmatrix,
            self.temp_pmatrix, self.display_mode)

        self.surface = pygame.Surface(self.size)
        psx3d = pygame.surfarray.pixels3d(self.surface)

        psx3d[:] = self.colors_array
        del psx3d

    def set_display_mode(self) -> None:
        self.display_mode = 1 if self.display_mode == 0 else 0

    def set_pause(self) -> None:
        self.pause = not self.pause

    def set_brush_mode(self) -> None:
        self.brush_mode = 1 if self.brush_mode == 0 else 0

    def select_pixel(self, pix_id: int) -> None:
        self.selected_pix = pix_id

    @staticmethod
    @njit(parallel=True, nogil=True, fastmath=True)
    def get_color_array(colors_array: numpy.ndarray, pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray,
                        mode: int) -> numpy.ndarray:
        shape = pmatrix.shape

        for x in prange(1, shape[0] - 1):
            for y in prange(1, shape[1] - 1):
                if mode == 0:
                    colors_array[x, y] = COLORS[pmatrix[x, y]]
                elif mode == 1:
                    colors_array[x, y] = temperature_to_color(round(temp_pmatrix[x, y]))
        return colors_array

    def __getitem__(self, key):
        return self.pmatrix[*key]

    def __setitem__(self, key, value):
        self.pmatrix[*key] = value

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def changing_temp(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray, x0: int, y0: int,
                      x1: int, y1: int, heat_coef: float) -> None:
        if check_borders(x0 + x1, y0 + y1, temp_pmatrix.shape[0], temp_pmatrix.shape[1]):
            additional_pix = pmatrix[x0 + x1, y0 + y1]

            if not SUBS_CHARS_PIXS[additional_pix, 3] or not additional_pix:
                return

            n0 = temp_pmatrix[x0, y0]
            n1 = temp_pmatrix[x0 + x1, y0 + y1]

            if n0 == n1:
                return

            dens1 = DENSITY[pmatrix[x0, y0]]
            dens2 = DENSITY[additional_pix]

            heat_cap1 = HEAT_CAPACITY[pmatrix[x0, y0]]
            heat_cap2 = HEAT_CAPACITY[additional_pix]

            diff_chars = (heat_cap1 / heat_cap2) / dens1 * dens2
            diff_chars = 1 / diff_chars if diff_chars > 1 else diff_chars

            diff = abs(n0 - (n0 + n1) / 2) * heat_coef * diff_chars

            temp_pmatrix[x0, y0] -= diff * (n0 > n1)
            temp_pmatrix[x0 + x1, y0 + y1] += diff * (n0 > n1)
            temp_pmatrix[x0, y0] += diff * (n0 < n1)
            temp_pmatrix[x0 + x1, y0 + y1] -= diff * (n0 < n1)
        else:
            temp_pmatrix[x0, y0] -= (temp_pmatrix[x0, y0] - DEFAULT_TEMPERATURE) / 1250 * heat_coef

    @staticmethod
    # @Utils.speedtest
    @njit(parallel=True, fastmath=True, nogil=True)
    def chem_iter(pmatrix, temp_pmatrix):
        shape_x, shape_y = temp_pmatrix.shape

        for x in prange(shape_x):
            for y in prange(shape_y):
                transformations = TEMPERATURE_TRANSFORMATIONS[pmatrix[x, y]]

                if transformations[0] == -666:
                    continue

                pixtemp = temp_pmatrix[x, y]

                if pixtemp < transformations[0]:
                    pmatrix[x, y] = transformations[1]
                elif pixtemp > transformations[2]:
                    pmatrix[x, y] = transformations[3]
                else:
                    pmatrix[x, y] = transformations[4]

    @staticmethod
    # @Utils.speedtest
    @njit(parallel=True, fastmath=True, nogil=True)
    def temp_iter(pmatrix, temp_pmatrix, heat_coef, heater_temp: int, cooler_temp: int):
        points = numpy.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
        # points = numpy.array([(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, 1)])
        ln = len(points)
        changing_temp = global_changing_temp
        shape_w, shape_h = temp_pmatrix.shape

        for x in prange(shape_w):
            for y in prange(shape_h):
                if pmatrix[x, y] == 15:
                    temp_pmatrix[x, y] = heater_temp
                    continue
                if pmatrix[x, y] == 16:
                    temp_pmatrix[x, y] = cooler_temp
                    continue

                if SUBS_CHARS_PIXS[pmatrix[x, y], 3] and pmatrix[x, y]:
                    for pind in prange(ln):
                        changing_temp(pmatrix, temp_pmatrix, x, y, points[pind][0], points[pind][1], heat_coef)

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def iter(pmatrix: numpy.ndarray, temp_pmatrix: numpy.ndarray):
        shape_x, shape_y = pmatrix.shape

        nine_tenths = 0.9
        one_twenty_fifth = 0.1

        for x_c in prange(1, shape_x - 1):
            x = shape_x - x_c - 1
            for y in prange(1, shape_y - 1):
                pixtype = pmatrix[x, y]
                pixtypes = SUBS_CHARS_PIXS[pixtype]

                if pixtypes[4]:
                    continue

                pixdensity = DENSITY[pixtype]

                if pixtypes[0]:
                    possibility = random()

                    x_n, y_n = x + 0, y + 1
                    if 0.85 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x - 1, y + 0
                    if 0.35 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x + 1, y + 0
                    if 0.15 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x - 1, y + 1
                    if 0.25 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x + 1, y + 1
                    if 0.1 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]
                elif pixtypes[1]:
                    possibility = random()

                    x_n, y_n = x, y + 1
                    if 0.95 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x + 1, y + 1
                    if 0.5 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]

                    x_n, y_n = x - 1, y + 1
                    if 0.5 > possibility and can_move(x_n, y_n, shape_x, shape_y, pixtype, pmatrix[x_n, y_n]):
                        pmatrix[x, y], pmatrix[x_n, y_n] = pmatrix[x_n, y_n], pmatrix[x, y]
                        temp_pmatrix[x, y], temp_pmatrix[x_n, y_n] = temp_pmatrix[x_n, y_n], temp_pmatrix[x, y]
                elif pixtypes[2]:
                    move_cond(x, y, x, y + 1, shape_x, shape_y, pixdensity, pmatrix, temp_pmatrix,
                              nine_tenths)
                    move_cond(x, y, x + 1, y, shape_x, shape_y, pixdensity, pmatrix, temp_pmatrix,
                              one_twenty_fifth)
                    move_cond(x, y, x - 1, y, shape_x, shape_y, pixdensity, pmatrix, temp_pmatrix,
                              one_twenty_fifth)

                    if y + 1 >= shape_y:
                        continue

                    left, right = 99999, -99999
                    x_cn = 0

                    while right == -99999 or left == 99999:
                        x_cn += 1

                        if right == -99999:
                            if check_borders(x + x_cn, y + 1, shape_x, shape_y):
                                if pixdensity > DENSITY[pmatrix[x + x_cn, y]] and not \
                                        SUBS_CHARS_PIXS[pmatrix[x + x_cn, y], 4]:
                                    if pixdensity > DENSITY[pmatrix[x + x_cn, y + 1]] and not \
                                            SUBS_CHARS_PIXS[pmatrix[x + x_cn, y + 1], 4]:
                                        right = x_cn
                                else:
                                    right = x_cn - 1
                            else:
                                right = x_cn - 1
                        if left == 99999:
                            if check_borders(x - x_cn, y + 1, shape_x, shape_y):
                                if pixdensity > DENSITY[pmatrix[x - x_cn, y]] and not \
                                        SUBS_CHARS_PIXS[pmatrix[x - x_cn, y], 4]:
                                    if pixdensity > DENSITY[pmatrix[x - x_cn, y + 1]] and not \
                                            SUBS_CHARS_PIXS[pmatrix[x - x_cn, y + 1], 4]:
                                        left = -x_cn
                                else:
                                    left = 1 - x_cn
                            else:
                                left = 1 - x_cn

                    if left == 99999 or right == -99999:
                        if left == 99999 and right == -99999:
                            selected_dir = False
                        elif left == 99999:
                            selected_dir = right
                        else:
                            selected_dir = left
                    else:
                        selected_dir = left if numpy.abs(left) > right else right

                    if selected_dir:
                        if pmatrix[x + selected_dir, y + 1] == 0:
                            pmatrix[x, y], pmatrix[x + selected_dir, y + 1] = pmatrix[x + selected_dir, y + 1], pmatrix[
                                x, y]
                            temp_pmatrix[x, y], temp_pmatrix[x + selected_dir, y + 1] = temp_pmatrix[
                                x + selected_dir, y + 1], temp_pmatrix[x, y]
                        else:
                            pmatrix[x, y], pmatrix[x + selected_dir, y] = pmatrix[x + selected_dir, y], pmatrix[x, y]
                            temp_pmatrix[x, y], temp_pmatrix[x + selected_dir, y] = temp_pmatrix[x + selected_dir, y], \
                                temp_pmatrix[x, y]


temperature_to_color = Utils.temperature_to_color
global_changing_temp = Matrix.changing_temp
# global_get_color_array_kernel = Matrix.get_color_array_kernel
