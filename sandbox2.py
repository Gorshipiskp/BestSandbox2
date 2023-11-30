import numpy

if __name__ == "__main__":
    import collections
    import time
    from numba import set_num_threads, get_num_threads
    import pygame
    import pygame_widgets
    from pygame_widgets.button import ButtonArray, Button
    from pygame_widgets.slider import Slider
    from pygame_widgets.textbox import TextBox
    from basesandbox2 import Matrix, pixs, Utils, NAMES, language, cfg, COLORS

    FPS = cfg["MAX_FPS"]
    MENU_WIDTH = 260
    WID, HEI = cfg["WIDTH"] + 1, cfg["HEIGHT"] + 1
    SCALE = cfg["SCALE"]
    BOOSTED = cfg['BOOSTED']
    SCRNSIZE = (WID, HEI)
    SCRNSIZE_SCALED = (WID * SCALE, HEI * SCALE)

    if not BOOSTED:
        set_num_threads(max((get_num_threads() - 1, 1)))

    widgetsClasses = {
        'Slider': {'class': Slider, 'size': (MENU_WIDTH - 40, 25)},
        'TextBox': {'class': TextBox, 'size': (MENU_WIDTH - 20, 25)},
        'Button': {'class': Button, 'size': (MENU_WIDTH - 20, 20)}
    }

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
        params = collections.deque()

        for pix_id, _ in enumerate(pixs):
            params.append((pix_id,))

        last = None
        pygame.init()
        STFONT = pygame.font.SysFont("Bahnschrift", 16)
        STFONT_SMALL = pygame.font.SysFont("Bahnschrift", 14)
        STFONT_BOLD_SMALL = pygame.font.SysFont("Bahnschrift", 14)
        STFONT_BOLD_SMALL.set_bold(True)

        screen = pygame.display.set_mode((WID * SCALE + MENU_WIDTH, HEI * SCALE))
        pygame.display.set_caption(language['other']['title'])

        clock = pygame.time.Clock()
        stopped = False
        last_speed = [1]
        game_speed = [1]

        def toggle_pause():
            if game_speed[0] != 0:
                last_speed[0] = game_speed[0]
                game_speed[0] = 0
                widgets['speed_slider']['ref'].setValue(0)
            else:
                game_speed[0] = last_speed[0]
                widgets['speed_slider']['ref'].setValue(last_speed[0])

        widgets = {
            'brush_size_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'brush_size_slider': {
                'type': "Slider", 'attrs': {'min': 1, 'max': 75, 'handleRadius': 11,
                                            'handleColour': (110, 115, 120), 'initial': 3}, "ExtPadding": True,
            },
            'brush_mode_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'brush_mode_button': {
                'type': "Button", 'attrs': {'text': language['other']['brshmd'], 'onClick': mtrx.set_brush_mode},
                "ExtPadding": True,
            },
            'heat_quan_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'heat_quan_slider': {
                'type': "Slider", 'attrs': {'min': 0, 'max': 200, 'handleRadius': 11,
                                            'handleColour': (110, 115, 120), 'initial': 110}, "ExtPadding": True,
            },
            'cell_temp_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT},
                "ExtPadding": True,
            },
            'display_mode_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'display_button': {
                'type': "Button", 'attrs': {'text': language['other']['dspmd'], 'onClick': mtrx.set_display_mode},
                "ExtPadding": True,
            },
            'heat_coef_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'heat_coef_slider': {
                'type': "Slider", 'attrs': {'min': 0, 'max': 1.5, 'handleRadius': 11, 'step': 0.01,
                                            'handleColour': (110, 115, 120), 'initial': 0.7}, "ExtPadding": True,
            },
            'heater_temp_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'heater_temp_slider': {
                'type': "Slider", 'attrs': {'min': 273.15, 'max': 2773.15, 'handleRadius': 11, 'step': 0.1,
                                            'handleColour': (110, 115, 120), 'initial': 773.15}, "ExtPadding": True,
            },
            'cooler_temp_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'cooler_temp_slider': {
                'type': "Slider", 'attrs': {'min': 0, 'max': 273.15, 'handleRadius': 11, 'step': 0.1,
                                            'handleColour': (110, 115, 120), 'initial': 173}, "ExtPadding": True,
            },
            'speed_textbox': {
                'type': "TextBox", 'attrs': {'borderThickness': 0, 'font': STFONT}
            },
            'speed_slider': {
                'type': "Slider", 'attrs': {'min': 0, 'max': 1, 'handleRadius': 11, 'step': 0.01,
                                            'handleColour': (110, 115, 120), 'initial': 1}, "ExtPadding": True,
            },
            'onpausebutton': {
                'type': "Button", 'attrs': {'text': language['other']['pause'], 'onClick': toggle_pause}
            },
            'reset_field_button': {
                'type': "Button", 'attrs': {'text': language['other']['reset_field'], 'onClick': mtrx.reset_field}
            },
        }
        padding = 5
        yp = 5
        extPad = 5

        for name_id, vals in widgets.items():
            size = widgetsClasses[vals['type']]['size']

            w_p = 10 * (vals['type'] == "Slider")

            vals['ref'] = widgetsClasses[vals['type']]['class'](screen, 10 + w_p, yp, *size, **vals['attrs'])

            if vals.get("selfParam"):
                vals['ref'].onClickParams = (vals['ref'],)

            yp += size[1] + padding
            if vals.get('ExtPadding'):
                yp += extPad

        ButtonArray(
            screen, 5, yp + padding, MENU_WIDTH - 10, 20 * len(pixs) - 15,
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
        speed_count = 0
        ctrl_pressed = False
        line_move = False

        # start_playing = time.time()

        Utils.drawline(mtrx.pmatrix, mtrx.temp_pmatrix, 0, 0, 0, 0, 1, mtrx.selected_pix,
                       mtrx.brush_mode, 1, *mtrx.size)

        # while not stopped and time.time() - start_playing <= 300:
        while not stopped:
            start = time.time()

            # print(numpy.sum(mtrx.temp_pmatrix[:, :]), numpy.max(mtrx.temp_pmatrix[:, :]),
            #       numpy.min(mtrx.temp_pmatrix[:, :]))

            clock.tick(FPS)

            BRUSH_SIZE = widgets['brush_size_slider']['ref'].getValue()
            heat_quan = widgets['heat_quan_slider']['ref'].getValue() - 100
            heat_coef = float(widgets['heat_coef_slider']['ref'].getValue())
            heater_temp = widgets['heater_temp_slider']['ref'].getValue()
            cooler_temp = widgets['cooler_temp_slider']['ref'].getValue()
            game_speed[0] = widgets['speed_slider']['ref'].getValue()

            widgets['brush_size_textbox']['ref'].text = f"{language['other']['brshsz']}: {BRUSH_SIZE}"
            widgets['display_mode_textbox']['ref'].text = \
                f"{language['other']['mode']}: {display_modes[mtrx.display_mode]}"
            widgets['brush_mode_textbox']['ref'].text = f"{language['other']['mode']}: {brush_modes[mtrx.brush_mode]}"
            widgets['heat_coef_textbox']['ref'].text = f"{language['other']['heat_coef']}: {heat_coef:.2f}"
            widgets['heater_temp_textbox']['ref'].text = \
                f"{language['other']['heater_temp']}: {heater_temp - 273.15:.2f}°C"
            widgets['cooler_temp_textbox']['ref'].text = \
                f"{language['other']['cooler_temp']}: {cooler_temp - 273.15:.2f}°C"
            widgets['speed_textbox']['ref'].text = f"{language['other']['speed']}: {game_speed[0]:.2f}"
            widgets['onpausebutton']['ref'].text = STFONT_SMALL.render(
                language['other']['start'] if game_speed[0] == 0 else language['other']['pause'], True, (5, 5, 5))
            pos = pygame.mouse.get_pos()
            pos_x, pos_y = (pos[0] - MENU_WIDTH) // SCALE, pos[1] // SCALE

            if 0 <= pos_x < mtrx.size[0] and 0 <= pos_y < mtrx.size[1]:
                # cell_temp_textbox.text = f"{mtrx.temp_pmatrix[pos_x, pos_y]:.2f}K"
                widgets['cell_temp_textbox']['ref'].text = f"{mtrx.temp_pmatrix[pos_x, pos_y] - 273.15:.2f}°C"
            else:
                widgets['cell_temp_textbox']['ref'].text = "-"

            widgets['heat_quan_textbox']['ref'].text = f"{language['other']['power']}: " \
                                                       f"{'+' if heat_quan > 0 else ''}{heat_quan}°C"

            speed_count += 1
            if game_speed[0] != 0 and speed_count % int(1 / game_speed[0]) == 0:

                mtrx.iter(mtrx.pmatrix, mtrx.temp_pmatrix)
                if heat_coef > 0:
                    mtrx.temp_iter(mtrx.pmatrix, mtrx.temp_pmatrix, heat_coef, heater_temp, cooler_temp)
                    mtrx.borders_cool(mtrx.pmatrix, mtrx.temp_pmatrix, heat_coef)
                mtrx.chem_iter(mtrx.pmatrix, mtrx.temp_pmatrix)

            if ctrl_pressed and line_move:
                mtrx_layer = numpy.full_like(mtrx.pmatrix, 0)
                Utils.drawline_layer(mtrx_layer, *line_move, pos_x, pos_y, BRUSH_SIZE - 1, mtrx.selected_pix)
            else:
                mtrx_layer = numpy.full_like(mtrx.pmatrix, 0)

            psx3d = pygame.surfarray.pixels3d(mtrx.surface)
            psx3d = mtrx.get_color_array(psx3d, mtrx.pmatrix, mtrx.temp_pmatrix, mtrx.display_mode, mtrx_layer)
            del psx3d

            if not (pygame.key.get_pressed()[pygame.K_LCTRL] and pygame.mouse.get_pressed()[0]) and ctrl_pressed:
                if line_move:
                    Utils.drawline(mtrx.pmatrix, mtrx.temp_pmatrix, *line_move, pos_x, pos_y, BRUSH_SIZE - 1,
                                   mtrx.selected_pix, mtrx.brush_mode, heat_quan, *mtrx.size)
                    line_move = False
                    ctrl_pressed = pygame.key.get_pressed()[pygame.K_LCTRL]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stopped = True
                if event.type == pygame.KEYDOWN:
                    if pygame.key.get_pressed()[pygame.K_SPACE]:
                        toggle_pause()
                    ctrl_pressed = pygame.key.get_pressed()[pygame.K_LCTRL]
                if pygame.mouse.get_pressed()[0]:
                    if pos_x >= 0 and pos_y >= 0:
                        x0, y0 = pos_x, pos_y

                        if ctrl_pressed:
                            if not line_move:
                                line_move = (x0, y0)
                        else:
                            if last:
                                x1, y1 = last
                            else:
                                x1, y1 = x0, y0

                            Utils.drawline(mtrx.pmatrix, mtrx.temp_pmatrix, x0, y0, x1, y1, BRUSH_SIZE - 1,
                                           mtrx.selected_pix, mtrx.brush_mode, heat_quan, *mtrx.size)
                            last = (pos_x, pos_y)
                else:
                    last = None

            pygame_widgets.update(pygame.event.get())

            if SCALE != 1:
                screen.blit(pygame.transform.scale(mtrx.surface, SCRNSIZE_SCALED), (MENU_WIDTH, 0))
            else:
                screen.blit(mtrx.surface, (MENU_WIDTH, 0), area=mtrx.surface.get_rect())
            pygame.display.update()

            eps_time = time.time() - start

            # if eps_time == 0:
            #     print(f"{eps_time:.8f}")
            # else:
            #     print(f"{eps_time:.8f} – {1 / eps_time}q/s")

            if eps_time != 0 and game_speed != 0:
                FPS_C_S += 1 / eps_time
                FPS_C_N += 1

        print(f"AVERAGE {FPS_C_S / FPS_C_N:.2f}FPS")


    if __name__ == "__main__":
        main()
        time.sleep(2)
