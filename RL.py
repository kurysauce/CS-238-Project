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
        """Simplify the board and piece information into a tuple for Q-learning."""
        board_state = tuple(tuple(1 if cell > 0 else 0 for cell in row) for row in tetris.board)
        piece_state = (tetris.figure.type, tetris.figure.rotation, tetris.figure.x, tetris.figure.y)
        return (board_state, piece_state)

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return random.choice(['left', 'right', 'down', 'rotate', 'drop'])
        else:
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

    def get_reward(self, tetris, initial_score):
        """Improved reward structure for line clears and compact play."""
        reward = 0

        # Reward for line clears (scaled significantly higher)
        score_diff = tetris.score - initial_score
        if score_diff > 0:
            reward += score_diff * 100  # Reward line clears heavily

            # Additional bonus for multiple line clears
            lines_cleared = score_diff // 100  # Assuming 100 points per line clear
            if lines_cleared > 1:
                reward += (lines_cleared ** 2) * 100  # Exponential bonus for multi-line clears

        # Penalize one-unit gaps
        one_unit_gaps = sum(
            1 for row in range(1, tetris.rows)
            for col in range(tetris.cols)
            if tetris.board[row][col] == 0 and tetris.board[row - 1][col] > 0
        )
        reward -= one_unit_gaps * 10  # Strong penalty for gaps

        # Penalize the height of the tallest column
        heights = [
            max(row for row in range(tetris.rows) if tetris.board[row][col] > 0)
            for col in range(tetris.cols)
            if any(tetris.board[row][col] > 0 for row in range(tetris.rows))
        ]
        max_height = max(heights, default=0)
        reward -= max_height * 3  # Increased penalty for tall columns

        # Encourage compact rows (fewer empty cells in a row)
        compactness_bonus = sum(
            tetris.cols - sum(1 for col in range(tetris.cols) if tetris.board[row][col] == 0)
            for row in range(tetris.rows)
        )
        reward += compactness_bonus * 0.2  # Small bonus for compact stacking

        # Game-over penalty
        if tetris.gameover:
            reward -= 2000  # Strong penalty for game over

        return reward
