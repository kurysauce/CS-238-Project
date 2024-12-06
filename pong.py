import gym
import numpy as np
import random
import math

# Hyperparameters
alpha = 0.01       # Smaller learning rate for stability
gamma = 0.99       # Discount factor
epsilon = 1.0      # Start with full exploration
epsilon_min = 0.001
epsilon_decay = 0.999  # Slower decay to allow more exploration
episodes = 20000   # Longer training period
max_timesteps = 1000
render_interval = 500  # Render every 500 episodes
initial_q_value = 1.0  # Optimistic initialization

# Actions: 0 (NOOP), 2 (UP), 3 (DOWN)
actions = [0, 2, 3]

# Q-table
Q = {}

# Discretization parameters
num_bins_x = 16
num_bins_y = 16
num_bins_v = 8  # For ball velocity

def create_bins(low, high, bins):
    return np.linspace(low, high, bins+1)

x_bins = create_bins(0, 160, num_bins_x)
y_bins = create_bins(0, 210, num_bins_y)
v_bins = create_bins(-20, 20, num_bins_v)  # Approximate velocity range

def discretize(value, bins):
    return np.clip(np.digitize(value, bins) - 1, 0, len(bins) - 1)

def extract_positions(observation):
    # White pixels are [236, 236, 236]
    white_pixels = np.where(np.all(observation == [236, 236, 236], axis=-1))
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

def get_discrete_state(observation, prev_ball_x=None, prev_ball_y=None):
    ball_x, ball_y, paddle_y = extract_positions(observation)
    if ball_x is None:
        ball_x = 80
    if ball_y is None:
        ball_y = 105
    if paddle_y is None:
        paddle_y = 105

    # Approximate ball velocity
    if prev_ball_x is not None and prev_ball_y is not None:
        ball_vx = ball_x - prev_ball_x
        ball_vy = ball_y - prev_ball_y
    else:
        ball_vx, ball_vy = 0, 0

    # Discretize positions and velocities
    bx = discretize(ball_x, x_bins)
    by = discretize(ball_y, y_bins)
    py = discretize(paddle_y, y_bins)
    bvx = discretize(ball_vx, v_bins)
    bvy = discretize(ball_vy, v_bins)

    return (bx, by, py, bvx, bvy), ball_x, ball_y

def choose_action(state):
    if state not in Q:
        Q[state] = np.ones(len(actions)) * initial_q_value  # Optimistic initialization

    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        return np.argmax(Q[state])

def update_q(state, action, reward, next_state):
    if next_state not in Q:
        Q[next_state] = np.ones(len(actions)) * initial_q_value
    old_value = Q[state][action]
    next_max = np.max(Q[next_state])
    Q[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

def ball_hits_paddle(observation):
    ball_x, ball_y, paddle_y = extract_positions(observation)
    return abs(ball_x - 144) < 10 and abs(ball_y - paddle_y) < 10
# Training loop
for episode in range(episodes):
    # Create environment with or without rendering
    if episode % render_interval == 0:
        env = gym.make('ALE/Pong-v5', render_mode='human')  # Render this episode
        print(f"Rendering episode {episode}...")
    else:
        env = gym.make('ALE/Pong-v5', render_mode=None)  # No rendering

    obs, info = env.reset()
    prev_ball_x, prev_ball_y = None, None
    state, prev_ball_x, prev_ball_y = get_discrete_state(obs)
    total_reward = 0

    for t in range(max_timesteps):
        action_idx = choose_action(state)
        action = actions[action_idx]
        next_obs, reward, done, truncated, info = env.step(action)
        next_state, next_ball_x, next_ball_y = get_discrete_state(next_obs, prev_ball_x, prev_ball_y)

        # Reward shaping: small reward for paddle-ball interaction
        if abs(next_ball_x - 144) < 10:  # Ball is close to the paddle
            reward += 50
        if ball_hits_paddle:
            reward += 100

        update_q(state, action_idx, reward, next_state)

        state = next_state
        prev_ball_x, prev_ball_y = next_ball_x, next_ball_y
        total_reward += reward

        if done or truncated:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Log progress
    if episode % 100 == 0:
        print(f"Episode {episode}/{episodes}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # Save Q-table for debugging or replication
    if episode == 100:
        with open("q_table_episode_100.npy", "wb") as f:
            np.save(f, Q)

    env.close()

print("Training complete.")
