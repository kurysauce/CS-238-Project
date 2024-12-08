import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import pickle
import csv

# Mute game audio
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998
learning_rate = 5e-5
batch_size = 32
memory_size = 100000
episodes = 150  # Adjust as needed
max_timesteps = 1000
target_update_interval = 1000

# Reward values
proximity_reward = 15
hit_paddle_reward = 30
score_reward = 75

# Environment setup
env = gym.make('ALE/Pong-v5', render_mode=None)
n_actions = env.action_space.n

# Replay buffer

replay_buffer = deque(maxlen=memory_size)

# Function to sample experiences
def sample_experiences():
    """Sample a batch of experiences from the replay buffer."""
    batch = random.sample(replay_buffer, batch_size)
    states = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array([exp[1] for exp in batch]), dtype=torch.long).to(device)
    rewards_batch = torch.tensor(np.array([exp[2] for exp in batch]), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array([exp[3] for exp in batch]), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array([exp[4] for exp in batch]), dtype=torch.float32).to(device)
    return states, actions, rewards_batch, next_states, dones


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

# Save Q-Table
def save_q_table(q_network, episode, filename):
    q_state_dict = q_network.state_dict()
    with open(f"{filename}_episode_{episode}.pkl", "wb") as f:
        pickle.dump(q_state_dict, f)
    print(f"Q-Table (weights) saved at episode {episode}.")

# Save metrics
def save_metrics_to_csv(metrics, filename):
    keys = metrics.keys()
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for i in range(len(metrics["episode"])):
            row = {key: metrics[key][i] for key in keys}
            writer.writerow(row)

# Preprocess and stack frames
def preprocess_frame(frame):
    frame = frame[34:194]
    frame = frame.mean(axis=2)
    frame = frame / 255.0
    frame = np.resize(frame, (84, 84))
    return frame

def stack_frames(stacked_frames, frame, is_new_episode):
    if is_new_episode:
        stacked_frames = np.stack([frame] * 4, axis=0)
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame
    return stacked_frames

# Training metrics
metrics = {"episode": [], "total_reward": [], "epsilon": []}
q_values_trend = []  # Track Q-values for graphing trends

# Log Q-values for the first 50 episodes
# Function to log Q-values
def log_q_values(q_network, episode, q_values_trend):
    q_state_dict = q_network.state_dict()
    layer_key = "fc.2.weight"  # Adjust based on your model architecture

    if layer_key in q_state_dict:
        q_values = q_state_dict[layer_key].cpu().detach().numpy()
        q_values_mean = np.mean(q_values)
        q_values_trend.append((episode, q_values_mean))
    else:
        print(f"Layer key '{layer_key}' not found in state_dict.")


# Training loop
for episode in range(episodes):
    obs, _ = env.reset()
    frame = preprocess_frame(obs)
    stacked_frames = stack_frames(None, frame, True)
    total_reward = 0.0

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
        # Adjust this based on your proximity_reward, hit_paddle_reward, etc.
        if reward == 1:
            reward += score_reward

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

        if t % target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if done or truncated:
            break

    # Log metrics and Q-values
    metrics["episode"].append(episode)
    metrics["total_reward"].append(total_reward)
    metrics["epsilon"].append(epsilon)

    if episode < 50:  # Save Q-values for the first 50 episodes
        save_q_table(q_network, episode, "q_table")
        log_q_values(q_network, episode, q_values_trend)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}/{episodes} completed with Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Save Q-values trends to a CSV
with open("q_values_trend.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Mean Q-Value"])
    writer.writerows(q_values_trend)

print("Q-values trends saved to 'q_values_trend.csv'.")

env.close()
