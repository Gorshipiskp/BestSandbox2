import collections
import time
import numpy
import pygame
import pygame_widgets
from basesandbox2 import Matrix, pixs, Utils
from pygame_widgets.button import ButtonArray
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

MENU_WIDTH = 200
WID, HEI = 1000, 750
SCALE = 1
SCRNSIZE = (WID, HEI)
SCRNSIZE_SCALED = (WID * SCALE, HEI * SCALE)

FPS = 75

mtrx = Matrix(SCRNSIZE)


def main():
    last = None
    pygame.init()
    STFONT = pygame.font.SysFont("Bahnschrift", 17)
    STFONT_BOLD_SMALL = pygame.font.SysFont("Bahnschrift", 14)
    STFONT_BOLD_SMALL.set_bold(True)

    screen = pygame.display.set_mode((WID * SCALE + MENU_WIDTH * SCALE, HEI * SCALE))
    pygame.display.set_caption("Sandbox")

    clock = pygame.time.Clock()
    stopped = False

    brush_size_slider: Slider = Slider(
        screen, 5, 5, MENU_WIDTH - 20, 25, min=1, max=25, handleRadius=11,
        handleColour=pygame.Color(100, 105, 110), initial=2
    )
    brush_size_textbox: TextBox = TextBox(screen, 5, 35, MENU_WIDTH - 20, 25, borderThickness=0, font=STFONT)
    brush_size_textbox.disable()

    names = collections.deque()
    params = collections.deque()

    for pix_id, vals in enumerate(pixs):
        names.append(vals['name'])
        params.append((pix_id,))

    ButtonArray(
        screen, 5, 65, MENU_WIDTH - 15, 20 * len(pixs),
        texts=names,
        fonts=[STFONT_BOLD_SMALL] * len(pixs),
        inactiveColour=(255, 0, 0),
        pressedColour=(0, 255, 0), radius=15,
        border=2,
        onClicks=[mtrx.select_pixel] * len(pixs),
        onClickParams=tuple(params),
        shape=(1, len(pixs))
    )

    while not stopped:
        colors = mtrx.get_color_array(mtrx.pmatrix, mtrx.size)
        pixelarray = pygame.surfarray.make_surface(colors)

        clock.tick(FPS)
        BRUSH_SIZE = brush_size_slider.getValue()
        brush_size_textbox.text = f"Размер кисточки: {BRUSH_SIZE}"
        mtrx.iter(mtrx.pmatrix)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stopped = True
            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                pos_x, pos_y = (pos[0] - MENU_WIDTH) // SCALE, pos[1] // SCALE

                if pos_x >= 0 and pos_y >= 0:
                    x0, y0 = pos_x, pos_y

                    if last:
                        x1, y1 = last[0], last[1]
                        pixels = tuple(Utils.get_pixels_on_circle(BRUSH_SIZE, x0, y0) for x0, y0 in
                                       Utils.get_line_points(x0, y0, x1, y1))

                        Utils.place_pixels_many(WID, HEI, mtrx.pmatrix, mtrx.selected_pix, pixels)
                    else:
                        Utils.place_pixels(WID, HEI, mtrx.pmatrix, mtrx.selected_pix,
                                           Utils.get_pixels_on_circle(BRUSH_SIZE), x0, y0)
                    last = (pos_x, pos_y)
            else:
                last = None

        pygame_widgets.update(pygame.event.get())
        screen.blit(pygame.transform.scale(pixelarray, SCRNSIZE_SCALED), (MENU_WIDTH, 0))
        pygame.display.update()


if __name__ == "__main__":
    main()
