import pygame
import random
from main import Tetris, ROWS, COLS, BLACK, WHITE, CELLSIZE, SCREEN, Assets, clock, FPS

# Initialize Pygame and create display
pygame.init()
print('gello')
win = pygame.display.set_mode(SCREEN)

# Initialize the Tetris game
tetris = Tetris(ROWS, COLS)
running = True
can_move = True  # Ensure the agent can move pieces

# Define the actions the agent can take
ACTIONS = ['left', 'right', 'down', 'rotate', 'drop']

while running:
    # Fill the background
    win.fill(BLACK)

    # Check if the game is over
    if tetris.gameover:
        print("Game Over! Final score:", tetris.score)
        tetris.__init__(ROWS, COLS)  # Restart the game if desired
        continue

    # Randomly select an action
    action = random.choice(ACTIONS)
    print(f"Agent action: {action}")  # Debugging statement to check actions

    # Execute the chosen action directly
    if action == 'left':
        tetris.go_side(-1)
    elif action == 'right':
        tetris.go_side(1)
    elif action == 'down':
        tetris.go_down()
    elif action == 'rotate':
        tetris.rotate()
    elif action == 'drop':
        tetris.go_space()

    # Render the game state
    for x in range(ROWS):
        for y in range(COLS):
            if tetris.board[x][y] > 0:
                val = tetris.board[x][y]
                img = Assets[val]
                win.blit(img, (y * CELLSIZE, x * CELLSIZE))
                pygame.draw.rect(win, WHITE, (y * CELLSIZE, x * CELLSIZE, CELLSIZE, CELLSIZE), 1)

    if tetris.figure:
        for i in range(4):
            for j in range(4):
                if i * 4 + j in tetris.figure.image():
                    img = Assets[tetris.figure.color]
                    x = CELLSIZE * (tetris.figure.x + j)
                    y = CELLSIZE * (tetris.figure.y + i)
                    win.blit(img, (x, y))
                    pygame.draw.rect(win, WHITE, (x, y, CELLSIZE, CELLSIZE), 1)

    # Update the display and maintain the frame rate
    pygame.display.update()
    clock.tick(FPS)

    # Clear any pending events to prevent manual input
    pygame.event.clear()  # Ensures no manual input affects the game
