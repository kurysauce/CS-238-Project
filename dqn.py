import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Mute game audio
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Hyperparameters
gamma = 0.99              # Discount factor
epsilon = 1.0             # Initial exploration rate
epsilon_min = 0.01        # Minimum epsilon
epsilon_decay = 0.995     # Epsilon decay
learning_rate = 1e-4      # Learning rate for the optimizer
batch_size = 32           # Batch size for experience replay
memory_size = 100000      # Replay buffer size
episodes = 20000          # Number of episodes
max_timesteps = 1000      # Max timesteps per episode
target_update_interval = 1000  # Update target network every 1000 steps

# Reward Shaping (Consider Adjusting These if Needed)
proximity_reward = 5
hit_paddle_reward = 10
score_reward = 20

# Environment setup (no rendering)
env = gym.make('ALE/Pong-v5', render_mode=None)
n_actions = env.action_space.n  # Number of actions

# Replay buffer
replay_buffer = deque(maxlen=memory_size)

# Neural network for approximating Q-values
class DQNetwork(nn.Module):
    def __init__(self, n_actions):
        super(DQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = DQNetwork(n_actions).to(device)
target_network = DQNetwork(n_actions).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

def preprocess_frame(frame):
    # Convert to grayscale, crop, resize, and normalize
    frame = frame[34:194]  # crop play area
    frame = frame.mean(axis=2)  # grayscale
    frame = frame / 255.0       # normalize
    frame = np.resize(frame, (84, 84))  # resize
    return frame

def stack_frames(stacked_frames, frame, is_new_episode):
    if is_new_episode:
        stacked_frames = np.stack([frame] * 4, axis=0)
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame
    return stacked_frames

def sample_experiences():
    minibatch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.int64, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
        torch.tensor(next_states, dtype=torch.float32, device=device),
        torch.tensor(dones, dtype=torch.float32, device=device),
    )

def ball_hits_paddle(ball_x, ball_y, paddle_y):
    # Approximate paddle position and size
    # This is a simplified heuristic, adjust if needed.
    paddle_height = 20  # approximate
    return abs(ball_x - 144) < 10 and abs(ball_y - paddle_y) < paddle_height / 2

def get_positions(obs):
    # Extract ball and paddle positions from RGB frame
    white_pixels = np.where(np.all(obs == [236, 236, 236], axis=-1))
    ys = white_pixels[0]
    xs = white_pixels[1]

    ball_x, ball_y = None, None
    paddle_y = None
    if len(xs) > 0:
        # Paddle detection
        paddle_mask = (xs == 144)
        paddle_ys = ys[paddle_mask]
        if len(paddle_ys) > 0:
            paddle_y = np.mean(paddle_ys)

        # Ball detection
        ball_mask = (xs > 16) & (xs < 144)
        ball_xs = xs[ball_mask]
        ball_ys = ys[ball_mask]

        if len(ball_xs) > 0:
            ball_x = np.mean(ball_xs)
            ball_y = np.mean(ball_ys)
    return ball_x, ball_y, paddle_y

for episode in range(episodes):
    obs, _ = env.reset()
    frame = preprocess_frame(obs)
    stacked_frames = stack_frames(None, frame, True)
    total_reward = 0.0

    # Extract initial positions for shaping
    prev_ball_x, prev_ball_y, prev_paddle_y = get_positions(obs)

    print(f"Starting Episode {episode + 1}/{episodes}")  # Print when episode starts

    for t in range(max_timesteps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_obs, reward, done, truncated, _ = env.step(action)
        next_frame = preprocess_frame(next_obs)
        next_stacked_frames = stack_frames(stacked_frames, next_frame, False)

        # Reward shaping
        ball_x, ball_y, paddle_y = get_positions(next_obs)
        if ball_x is not None and ball_y is not None and paddle_y is not None:
            # Proximity reward
            if abs(ball_x - 144) < 10:
                reward += proximity_reward
            # Paddle hit reward
            if prev_ball_x is not None and prev_ball_y is not None and prev_paddle_y is not None:
                if ball_hits_paddle(ball_x, ball_y, paddle_y):
                    reward += hit_paddle_reward
            # If base reward is +1 for scoring
            if reward == 1:
                reward += score_reward

        # Store experience
        replay_buffer.append((stacked_frames, action, reward, next_stacked_frames, done))

        stacked_frames = next_stacked_frames
        total_reward += reward

        # Training step
        if len(replay_buffer) > batch_size:
            states, actions, rewards_batch, next_states, dones = sample_experiences()
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0]
                targets = rewards_batch + (gamma * next_q_values * (1 - dones))

            q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if t % target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        prev_ball_x, prev_ball_y, prev_paddle_y = ball_x, ball_y, paddle_y

        if done or truncated:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print the total reward after the episode completes
    print(f"Episode {episode + 1}/{episodes} completed with Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()
