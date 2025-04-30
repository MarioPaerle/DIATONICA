import pygame
import numpy as np
import random as rd
import joblib

pygame.init()
display = pygame.display.set_mode((100*8, 50*8))

grid = np.zeros((100, 50))
scale = 8

def get_pixel_circle(center, radius):
    x0, y0 = center
    x = radius
    y = 0
    err = 0
    positions = []

    while x >= y:
        # Add all 8 symmetric points (octants)
        positions.append((x0 + x, y0 + y))
        positions.append((x0 + y, y0 + x))
        positions.append((x0 - y, y0 + x))
        positions.append((x0 - x, y0 + y))
        positions.append((x0 - x, y0 - y))
        positions.append((x0 - y, y0 - x))
        positions.append((x0 + y, y0 - x))
        positions.append((x0 + x, y0 - y))

        y += 1
        err += 1 + 2*y
        if 2*(err - x) + 1 > 0:
            x -= 1
            err += 1 - 2*x

    # Remove duplicates (if any) and return
    return list(set(positions))

def get_filled_pixel_circle(center, radius):
    x0, y0 = center
    positions = []
    r_squared = radius ** 2

    for dy in range(-radius, radius + 1):
        dx = int((r_squared - dy**2) ** 0.5)
        for x in range(x0 - dx, x0 + dx + 1):
            positions.append((x, y0 + dy))

    return positions

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    display.fill((0, 0, 0))
    for i, row in enumerate(grid):
        for j, case in enumerate(grid[i]):
            if case == 1:
                pygame.draw.circle(display, (255, 255, 255), (i*scale, j*scale), scale//2)

            if pygame.mouse.get_pressed()[0]:
                positions = get_filled_pixel_circle((xm, ym), 4)
                for position in positions:
                    try:
                        grid[position[0]][position[1]] = rd.randint(0, 1)
                    except IndexError:
                        pass
            elif pygame.mouse.get_pressed()[2]:
                positions = get_filled_pixel_circle((xm, ym), 4)
                for position in positions:
                    try:
                        grid[position[0]][position[1]] = 0
                    except IndexError:
                        pass

            elif pygame.mouse.get_pressed()[1]:
                name = input('enter file name >>>  ')
                joblib.dump(grid, f'{name}.pkl')
                print('saved!')

    mouse = pygame.mouse.get_pos()
    xm, ym = mouse[0] // scale, mouse[1] // scale

    pygame.display.flip()
