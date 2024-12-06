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
        # Column heights
        column_heights = [
            max((row for row in range(tetris.rows) if tetris.board[row][col] > 0), default=0)
            for col in range(tetris.cols)
        ]

        # Aggregate height
        aggregate_height = sum(column_heights)

        # Number of holes
        holes = sum(
            1 for col in range(tetris.cols)
            for row in range(column_heights[col])
            if tetris.board[row][col] == 0
        )

        # Bumpiness
        bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))

        # Rows cleared (last move)
        lines_cleared = tetris.remove_line()

        # Combine features into a tuple
        return (tuple(column_heights), aggregate_height, holes, bumpiness, lines_cleared)



    def choose_action(self, state):
        """Choose an action with a bias toward exploration initially."""
        if np.random.rand() < self.epsilon:  # Exploration
            return random.choice(['left', 'right', 'down', 'rotate', 'drop'])
        else:  # Exploitation
            q_values = {a: self.Q_table[(state, a)] for a in ['left', 'right', 'down', 'rotate', 'drop']}
            return max(q_values, key=q_values.get)



    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the state-action pair using the Q-learning formula."""
        max_future_q = max(self.Q_table[(next_state, a)] for a in ['left', 'right', 'down', 'rotate', 'drop'])
        current_q = self.Q_table[(state, action)]
        self.Q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
        """Reduce epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def get_reward(self, tetris):
        """Reward based on gameplay performance and features."""
        reward = 0

        # Reward for lines cleared
        lines_cleared = tetris.remove_line()
        reward += lines_cleared * 100

        # Penalize for high columns
        heights = [max((row for row in range(tetris.rows) if tetris.board[row][col] > 0), default=0) for col in range(tetris.cols)]
        total_height = sum(heights)
        reward -= total_height * 0.5

        # Penalize for holes
        holes = sum(
            1
            for col in range(tetris.cols)
            for row in range(1, tetris.rows)
            if tetris.board[row][col] == 0 and tetris.board[row - 1][col] > 0
        )
        reward -= holes * 1.5

        # Penalize for bumpiness
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        reward -= bumpiness * 0.5

        # Game over penalty
        if tetris.gameover:
            reward -= 500

        return reward



