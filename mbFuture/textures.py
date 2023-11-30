from os import listdir

import numba
from PIL import Image
from numpy import array, ndarray


def process_textures(dir_t: str = "textures", ext_fl: str = 'png'):
    textures_files = [i.removesuffix(f".{ext_fl}") for i in listdir(dir_t)]
    textures: dict[str: ndarray] = {}

    for texture in textures_files:
        textures[texture] = get_pixels_array(Image.open(f"{dir_t}/{texture}.{ext_fl}", mode='r'))
    return textures


def get_pixels_array(img: Image) -> ndarray:
    return array(img.convert('RGB').getdata()).reshape((*img.size, 3))


@numba.njit(fastmath=True, cache=True)
def get_pixtex_color(texture: ndarray, x: int, y: int) -> list[int, int, int]:
    return texture[x % texture.shape[0], y % texture.shape[1]]
