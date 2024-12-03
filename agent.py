import numpy as np
import pygame
import random
import time
from RL import QLearningAgent  # Import the Q-learning agent

pygame.init()
SCREEN = WIDTH, HEIGHT = 300, 500
win = pygame.display.set_mode(SCREEN, pygame.NOFRAME)

CELLSIZE = 20
ROWS = (HEIGHT-120) // CELLSIZE
COLS = WIDTH // CELLSIZE

clock = pygame.time.Clock()
FPS = 2000

# COLORS *********************************************************************
BLACK = (21, 24, 29)
BLUE = (31, 25, 76)
RED = (252, 91, 122)
WHITE = (255, 255, 255)

# Load assets
img1 = pygame.image.load('Assets/1.png')
img2 = pygame.image.load('Assets/2.png')
img3 = pygame.image.load('Assets/3.png')
img4 = pygame.image.load('Assets/4.png')

Assets = {1: img1, 2: img2, 3: img3, 4: img4}

# Fonts
font = pygame.font.Font('Fonts/Alternity-8w7J.ttf', 50)
font2 = pygame.font.SysFont('cursive', 25)

# Tetris shapes
class Tetramino:
    FIGURES = {
        'I': [[1, 5, 9, 13], [4, 5, 6, 7]],
        'Z': [[4, 5, 9, 10], [2, 6, 5, 9]],
        'S': [[6, 7, 9, 10], [1, 5, 6, 10]],
        'L': [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        'J': [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        'T': [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        'O': [[1, 2, 5, 6]]
    }
    TYPES = ['I', 'Z', 'S', 'L', 'J', 'T', 'O']

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.choice(self.TYPES)
        self.shape = self.FIGURES[self.type]
        self.color = random.randint(1, 4)
        self.rotation = 0

    def image(self):
        return self.shape[self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)

# Tetris game class
class Tetris:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.score = 0
        self.level = 1
        self.board = [[0 for j in range(cols)] for i in range(rows)]
        self.next = None
        self.gameover = False
        self.new_figure()

    def draw_grid(self):
        for i in range(self.rows+1):
            pygame.draw.line(win, WHITE, (0, CELLSIZE*i), (WIDTH, CELLSIZE*i))
        for j in range(self.cols):
            pygame.draw.line(win, WHITE, (CELLSIZE*j, 0), (CELLSIZE*j, HEIGHT-120))

    def new_figure(self):
        if not self.next:
            self.next = Tetramino(5, 0)
        self.figure = self.next
        self.next = Tetramino(5, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.rows - 1 or \
                       j + self.figure.x > self.cols - 1 or \
                       j + self.figure.x < 0 or \
                       self.board[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def remove_line(self):
        rerun = False
        for y in range(self.rows-1, 0, -1):
            is_full = True
            for x in range(0, self.cols):
                if self.board[y][x] == 0:
                    is_full = False
            if is_full:
                del self.board[y]
                self.board.insert(0, [0 for i in range(self.cols)])
                self.score += 1
                if self.score % 10 == 0:
                    self.level += 1
                rerun = True

        if rerun:
            self.remove_line()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.board[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.remove_line()
        self.new_figure()
        if self.intersects():
            self.gameover = True

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def go_side(self, dx):
        self.figure.x += dx
        if self.intersects():
            self.figure.x -= dx

    def rotate(self):
        rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = rotation
# Training loop for Q-Learning Agent
agent = QLearningAgent(alpha=0.5, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, num_episodes=10000, max_steps=100)
use_random_agent = False  # Toggle between random agent and Q-learning agent
training_episodes = 5000  # Number of episodes for training
# Inside the training loop

# Training loop
for episode in range(training_episodes):
    tetris = Tetris(ROWS, COLS)  # Reset the game environment for each episode
    initial_score = tetris.score
    total_reward = 0
    dropped_piece_count = 0  # Initialize dropped piece counter for the episode
    running = True

    while running:
        # Check if the game is over
        if tetris.gameover:
            print(f"Game Over! Score: {tetris.score}")
            break

        # Choose an action
        current_state = agent.get_state(tetris)
        action = agent.choose_action(current_state)

        # Perform the selected action
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
            dropped_piece_count += 1  # Increment when a piece is hard-dropped

        # Automatically move tetromino down
        tetris.go_down()
        if tetris.figure.y == tetris.rows - 1 or tetris.intersects():
            dropped_piece_count += 1  # Increment for a natural drop that places a piece

        # Compute reward and update Q-table
        reward = agent.get_reward(tetris, dropped_piece_count)  # Pass dropped_piece_count to get_reward
        next_state = agent.get_state(tetris)
        agent.update_q_value(current_state, action, reward, next_state)

        # Accumulate total reward for the episode
        total_reward += reward

        # Render game elements
        win.fill(BLACK)  # Clear the screen

        # Draw the board
        for x in range(ROWS):
            for y in range(COLS):
                if tetris.board[x][y] > 0:
                    val = tetris.board[x][y]
                    img = Assets[val]
                    win.blit(img, (y * CELLSIZE, x * CELLSIZE))
                    pygame.draw.rect(win, WHITE, (y * CELLSIZE, x * CELLSIZE, CELLSIZE, CELLSIZE), 1)

        # Draw the current tetromino
        if tetris.figure:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in tetris.figure.image():
                        img = Assets[tetris.figure.color]
                        x = CELLSIZE * (tetris.figure.x + j)
                        y = CELLSIZE * (tetris.figure.y + i)
                        win.blit(img, (x, y))
                        pygame.draw.rect(win, WHITE, (x, y, CELLSIZE, CELLSIZE), 1)

        # Draw the HUD
        pygame.draw.rect(win, BLUE, (0, HEIGHT - 120, WIDTH, 120))
        if tetris.next:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in tetris.next.image():
                        img = Assets[tetris.next.color]
                        x = CELLSIZE * (tetris.next.x + j - 4)
                        y = HEIGHT - 100 + CELLSIZE * (tetris.next.y + i)
                        win.blit(img, (x, y))

        # Render score and level
        scoreimg = font.render(f'{tetris.score}', True, WHITE)
        levelimg = font2.render(f'Level : {tetris.level}', True, WHITE)
        win.blit(scoreimg, (250 - scoreimg.get_width() // 2, HEIGHT - 110))
        win.blit(levelimg, (250 - levelimg.get_width() // 2, HEIGHT - 30))

        pygame.draw.rect(win, BLUE, (0, 0, WIDTH, HEIGHT - 120), 2)

        # Refresh the display
        clock.tick(FPS)
        pygame.display.update()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()

    # Decay epsilon for exploration-exploitation tradeoff
    agent.decay_epsilon()

    # Log progress
    if episode % 100 == 0:
        print(f"Episode {episode}/{training_episodes}, Total Reward: {total_reward}, Dropped Pieces: {dropped_piece_count}")
