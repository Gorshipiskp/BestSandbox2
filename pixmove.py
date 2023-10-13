from random import random
import numpy
from numba import njit


@njit(fastmath=True, cache=True)
def check_borders(x: int, y: int, shape: tuple[int, int] | numpy.ndarray) -> bool:
    return 0 <= x < shape[0] and 0 <= y < shape[1]


@njit(fastmath=True, cache=True)
def move_cond(x0: int, y0: int, x1: int, y1: int, shape, pixdensity: float, pmatrix: numpy.ndarray,
              pmatrix_bools: numpy.ndarray, density: numpy.ndarray, possibility: float, ret_posdens_bool: bool = False):
    if check_borders(x1, y1, shape):
        if density[pmatrix[x1, y1, 0]] < pixdensity:
            if random() < possibility:
                temp = pmatrix[x1, y1].copy()
                pmatrix[x1, y1] = pmatrix[x0, y0]
                pmatrix[x0, y0] = temp
                pmatrix_bools[x0, y0] = True
                pmatrix_bools[x1, y1] = True
    if ret_posdens_bool:
        return check_borders(x1, y1, shape) and not density[pmatrix[x1, y1, 0]] < pixdensity
