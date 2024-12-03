import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.9, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, num_episodes=10000, max_steps=100):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.num_episodes = num_episodes  # Number of times agent will play game
        self.max_steps = max_steps 
        self.Q_table = defaultdict(float)  # Q-table as a dictionary

    def get_state(self, tetris):
        """Include spatial features like column heights and variance in state representation."""
        board_state = tuple(tuple(1 if cell > 0 else 0 for cell in row) for row in tetris.board)
        column_heights = [
            max(row for row in range(tetris.rows) if tetris.board[row][col] > 0)
            if any(tetris.board[row][col] > 0 for row in range(tetris.rows)) else 0
            for col in range(tetris.cols)
        ]
        height_variance = np.var(column_heights)
        return (board_state, tuple(column_heights), height_variance)


    def choose_action(self, state):
        """Choose an action with a bias toward exploring the board."""
        if np.random.rand() < self.epsilon:  # Exploration
            # Biased random choice to prioritize left/right
            actions = ['left', 'right', 'down', 'rotate', 'drop']
            weights = [0.4, 0.4, 0.1, 0.05, 0.05]  # Favor left and right
            return random.choices(actions, weights=weights, k=1)[0]
        else:  # Exploitation
            q_values = {a: self.Q_table[(state, a)] for a in ['left', 'right', 'down', 'rotate', 'drop']}
            return max(q_values, key=q_values.get)


    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the state-action pair using the Q-learning formula."""
        max_future_q = max(self.Q_table[(next_state, a)] for a in ['left', 'right', 'down', 'rotate', 'drop'])
        current_q = self.Q_table[(state, action)]
        self.Q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_reward(self, tetris, dropped_piece_count):
        """Calculate the reward based on gameplay metrics."""
        reward = 0
        
        # 1 point per dropped piece
        reward += dropped_piece_count

        # Reward for clearing lines: board_width * (lines_cleared^2)
        lines_cleared = tetris.score - dropped_piece_count  # Assuming score increments by lines cleared
        board_width = tetris.cols
        reward += board_width * (lines_cleared ** 2)

        # Calculate the number of holes (empty cells with a filled cell above them)
        holes = sum(
            1 for col in range(tetris.cols)
            for row in range(1, tetris.rows)
            if tetris.board[row][col] == 0 and tetris.board[row - 1][col] > 0
        )
        reward -= holes * 2  # Penalize heavily for holes

        # Calculate the bumpiness (difference in heights between adjacent columns)
        column_heights = [
            max((row for row in range(tetris.rows) if tetris.board[row][col] > 0), default=0)
            for col in range(tetris.cols)
        ]
        bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))
        reward -= bumpiness * 0.5  # Penalize uneven surfaces

        # Sum of column heights (to discourage tall stacks)
        total_height = sum(column_heights)
        reward -= total_height * 0.1  # Penalize high total height

        # Game over penalty
        if tetris.gameover:
            reward -= 1  # Strong penalty for losing

        return reward



