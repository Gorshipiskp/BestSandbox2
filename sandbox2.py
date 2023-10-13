import collections
import time
import numba
import pygame
import numpy
import pygame_widgets
from basesandbox2 import Matrix, pixs, Utils, NAMES, language, DEFAULT_STATUSES
from pygame_widgets.button import ButtonArray, Button
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from json import load


cfg = load(open('config.json', 'r', encoding="UTF-8"))


FPS = cfg["MAX_FPS"]
MENU_WIDTH = 215
WID, HEI = cfg["WIDTH"], cfg["HEIGHT"]
SCALE = cfg["SCALE"]
SCRNSIZE = (WID, HEI)
SCRNSIZE_SCALED = (WID * SCALE, HEI * SCALE)


mtrx = Matrix(SCRNSIZE)

display_modes = {
    0: language['display_modes']['nrml'],
    1: language['display_modes']['temp'],
}

brush_modes = {
    0: language['brush_modes']['subs'],
    1: language['brush_modes']['heat'],
}


def main():
    last = None
    pygame.init()
    STFONT = pygame.font.SysFont("Bahnschrift", 17)
    STFONT_SMALL = pygame.font.SysFont("Bahnschrift", 14)
    STFONT_BOLD_SMALL = pygame.font.SysFont("Bahnschrift", 14)
    STFONT_BOLD_SMALL.set_bold(True)

    screen = pygame.display.set_mode((WID * SCALE + MENU_WIDTH, HEI * SCALE))
    pygame.display.set_caption(language['other']['title'])

    clock = pygame.time.Clock()
    stopped = False

    brush_size_slider: Slider = Slider(
        screen, 10, 5, MENU_WIDTH - 30, 25, min=1, max=25, handleRadius=11,
        handleColour=pygame.Color(110, 115, 120), initial=3
    )
    brush_size_textbox: TextBox = TextBox(screen, 5, 35, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    brush_size_textbox.disable()

    def toggle_pause(btn):
        mtrx.set_pause()
        btn.text = STFONT_SMALL.render(language['other']['strt'] if mtrx.pause else language['other']['pause'], True, (5, 5, 5))

    params = collections.deque()

    for pix_id, _ in enumerate(pixs):
        params.append((pix_id,))

    Button(screen, 10, 75, MENU_WIDTH - 10, 20, text=language['other']['brshmd'], onClick=mtrx.set_brush_mode)
    brush_mode_textbox: TextBox = TextBox(screen, 5, 100, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    brush_mode_textbox.disable()

    heat_quan_slider: Slider = Slider(
        screen, 5, 130, MENU_WIDTH - 30, 25, min=0, max=200, handleRadius=11,
        handleColour=pygame.Color(110, 115, 120), initial=110
    )
    heat_quan_textbox: TextBox = TextBox(screen, 5, 160, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    heat_quan_textbox.disable()

    cell_temp_textbox: TextBox = TextBox(screen, 5, 195, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    cell_temp_textbox.disable()

    Button(screen, 5, 255, MENU_WIDTH - 10, 20, text=language['other']['dspmd'], onClick=mtrx.set_display_mode)
    display_mode_textbox: TextBox = TextBox(screen, 5, 225, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    display_mode_textbox.disable()

    onpausebutton = Button(screen, 5, 285, MENU_WIDTH - 10, 20, text=language['other']['pause'], onClick=toggle_pause)
    onpausebutton.onClickParams = (onpausebutton, )
    display_mode_textbox: TextBox = TextBox(screen, 5, 225, MENU_WIDTH - 15, 25, borderThickness=0, font=STFONT)
    display_mode_textbox.disable()

    ButtonArray(
        screen, 5, 315, MENU_WIDTH - 10, 20 * len(pixs) - 10,
        texts=NAMES,
        fonts=[STFONT_BOLD_SMALL] * len(pixs),
        inactiveColour=(255, 0, 0),
        pressedColour=(0, 255, 0), radius=15,
        border=2,
        onClicks=[mtrx.select_pixel] * len(pixs),
        onClickParams=tuple(params),
        shape=(1, len(pixs))
    )

    FPS_C_S = 0
    FPS_C_N = 0

    start_playing = time.time()

    x0, y0 = ((WID // 2) - MENU_WIDTH) // SCALE, (HEI // 2) // SCALE

    x1, y1 = 2, 2
    pixels = list(Utils.get_pixels_on_circle(5, x0, y0) for x0, y0 in
                  Utils.get_line_points(x0, y0, x1, y1))

    if pixels:
        Utils.place_pixels_many(mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.colors_array_bool, mtrx.selected_pix,
                                mtrx.brush_mode, mtrx.heat_quan,
                                numba.typed.List(pixels), DEFAULT_STATUSES)
    Utils.place_pixels(mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.selected_pix, mtrx.brush_mode, mtrx.heat_quan,
                       Utils.get_pixels_on_circle(5), DEFAULT_STATUSES, x0, y0)

    while not stopped and time.time() - start_playing <= 300:
        start = time.time()

        # print(numpy.sum(mtrx.temp_pmatrix[:, :]), numpy.max(mtrx.temp_pmatrix[:, :]),
        #       numpy.min(mtrx.temp_pmatrix[:, :]))

        clock.tick(FPS)
        psx3d = pygame.surfarray.pixels3d(mtrx.surface)
        psx3d, mtrx.colors_array_bool = mtrx.get_color_array(psx3d, mtrx.colors_array_bool, mtrx.pmatrix,
                                                             mtrx.temp_pmatrix, mtrx.display_mode)
        del psx3d

        heat_quan = heat_quan_slider.getValue() - 100

        mtrx.heat_quan = heat_quan

        BRUSH_SIZE = brush_size_slider.getValue()
        brush_size_textbox.text = f"{language['other']['brshsz']}: {BRUSH_SIZE}"
        display_mode_textbox.text = f"{language['other']['mode']}: {display_modes[mtrx.display_mode]}"
        brush_mode_textbox.text = f"{language['other']['mode']}: {brush_modes[mtrx.brush_mode]}"

        pos = pygame.mouse.get_pos()
        pos_x, pos_y = (pos[0] - MENU_WIDTH) // SCALE, pos[1] // SCALE

        if 0 <= pos_x < mtrx.size[0] and 0 <= pos_y < mtrx.size[1]:
            # cell_temp_textbox.text = f"{mtrx.temp_pmatrix[pos_x, pos_y]:.2f}K"
            cell_temp_textbox.text = f"{mtrx.temp_pmatrix[pos_x, pos_y] - 273.15:.2f}°C"

        heat_quan_textbox.text = f"{'+' if heat_quan > 0 else ''}{heat_quan}°C"

        if not mtrx.pause:
            mtrx.colors_array_bool = mtrx.iter(mtrx.colors_array_bool, mtrx.pmatrix)
            mtrx.temp_iter(mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.colors_array_bool)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stopped = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                toggle_pause(onpausebutton)
            elif pygame.mouse.get_pressed()[0]:
                if pos_x >= 0 and pos_y >= 0:
                    x0, y0 = pos_x, pos_y

                    if last:
                        x1, y1 = last[0], last[1]
                        pixels = list(Utils.get_pixels_on_circle(BRUSH_SIZE, x0, y0) for x0, y0 in
                                      Utils.get_line_points(x0, y0, x1, y1))

                        if pixels:
                            Utils.place_pixels_many(mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.colors_array_bool, mtrx.selected_pix,
                                                    mtrx.brush_mode, mtrx.heat_quan,
                                                    numba.typed.List(pixels), DEFAULT_STATUSES)
                    else:
                        Utils.place_pixels(mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.selected_pix, mtrx.brush_mode, mtrx.heat_quan,
                                           Utils.get_pixels_on_circle(BRUSH_SIZE), DEFAULT_STATUSES, x0, y0)
                    last = (pos_x, pos_y)
            else:
                last = None

        pygame_widgets.update(pygame.event.get())

        if SCALE != 1:
            screen.blit(pygame.transform.scale(mtrx.surface, SCRNSIZE_SCALED), (MENU_WIDTH, 0))
        else:
            screen.blit(mtrx.surface, (MENU_WIDTH, 0))
        pygame.display.update()

        eps_time = time.time() - start

        # if eps_time == 0:
        #     print(f"{eps_time:.8f}")
        # else:
        #     print(f"{eps_time:.8f} – {1 / eps_time}q/s")

        if eps_time != 0:
            FPS_C_S += 1 / eps_time
            FPS_C_N += 1

    print(f"AVERAGE {FPS_C_S / FPS_C_N:.2f}FPS")


if __name__ == "__main__":
    main()
