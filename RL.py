import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
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
        """Calculate a more complex reward based on line clears, gaps, and height."""
        reward = 0
        
        # Reward for score increase (i.e., lines cleared)
        score_diff = tetris.score - initial_score
        if score_diff > 0:
            reward += score_diff * 10  # Adjust this scaling factor as needed

        # Penalize for one-unit gaps
        one_unit_gaps = sum(
            1 for row in range(1, tetris.rows)
            for col in range(tetris.cols)
            if tetris.board[row][col] == 0 and tetris.board[row - 1][col] > 0
        )
        reward -= one_unit_gaps * 2  # Adjust penalty as needed

        # Penalize the height of the tallest column
        filled_rows = [row for row in range(tetris.rows) if any(tetris.board[row][col] > 0 for col in range(tetris.cols))]
        max_height = max(filled_rows) if filled_rows else 0
        reward -= max_height * 0.5  # Adjust scaling factor as needed

        # Encourage compactness (fewer gaps per row)
        row_gaps = sum(
            sum(1 for col in range(tetris.cols) if tetris.board[row][col] == 0)
            for row in range(tetris.rows)
        )
        reward -= row_gaps * 0.1  # Adjust penalty as needed

        # Large negative reward if game over
        if tetris.gameover:
            reward -= 100

        return reward
