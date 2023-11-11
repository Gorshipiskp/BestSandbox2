from random import random
import numpy
from numba import njit


@njit(fastmath=True, cache=True)
def check_borders(x: int, y: int, shape: tuple[int, int]) -> bool:
    return 0 <= x < shape[0] and 0 <= y < shape[1]


@njit(fastmath=True)
def move_cond(x0: int, y0: int, x1: int, y1: int, shape, pixdensity: float, pmatrix: numpy.ndarray,
              pmatrix_bools: numpy.ndarray, temp_pmatrix: numpy.ndarray, density: numpy.ndarray, possibility: float,
              ret_posdens_bool: bool = False):
    if check_borders(x1, y1, shape):
        if density[pmatrix[x1, y1]] < pixdensity:
            if random() < possibility:
                pmatrix[x0, y0], pmatrix[x1, y1] = pmatrix[x1, y1], pmatrix[x0, y0]
                temp_pmatrix[x1, y1], temp_pmatrix[x0, y0] = temp_pmatrix[x0, y0], temp_pmatrix[x1, y1]
                pmatrix_bools[x0, y0], pmatrix_bools[x1, y1] = True, True
    if ret_posdens_bool:
        return check_borders(x1, y1, shape) and not density[pmatrix[x1, y1]] < pixdensity
