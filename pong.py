import gym
import numpy as np
import random
# import matplotlib.pyplot as plt

# Create the Pong environment
env = gym.make('ALE/Pong-v5', render_mode='human')

# Q-learning hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000  # Number of episodes to train
max_timesteps = 500  # Max steps per episode

# Discretize the action space (Pong has 6 discrete actions)
actions = [0, 1, 2, 3, 4, 5]  # The discrete action space of Pong

# Initialize Q-table: A simple lookup table for actions
# For Pong, we can approximate the states using a simpler representation (e.g., just actions)
Q = {}

# Function to choose an action using epsilon-greedy strategy
def choose_action(state):
    # Convert state to a tuple to make it hashable (if it's not already a tuple)
    state = tuple(state.flatten())  # Flatten the state (in case it's an array)
    
    if random.uniform(0, 1) < epsilon:
        # Exploration: Random action
        return random.choice(actions)
    else:
        # Exploitation: Choose the action with highest Q-value for the current state
        return np.argmax(Q.get(state, np.zeros(len(actions))))

# Initialize the Q-table for all states
def initialize_q_table():
    for state in actions:  # Q-table for each action, simplistic approach for this demo
        Q[state] = np.zeros(len(actions))

# Reset the environment
def reset_environment():
    return env.reset()

# Train the agent
initialize_q_table()

# Training loop
for episode in range(episodes):
    state, info = reset_environment()
    total_reward = 0
    
    for t in range(max_timesteps):
        # Choose an action using epsilon-greedy
        action = choose_action(state)

        # Take the chosen action in the environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Convert the next_state to a tuple (hashable)
        next_state = tuple(next_state.flatten())
        
        # Update Q-value
        old_q_value = Q.get(state, np.zeros(len(actions)))[action]
        next_q_value = np.max(Q.get(next_state, np.zeros(len(actions))))
        
        Q[state][action] = old_q_value + alpha * (reward + gamma * next_q_value - old_q_value)
        
        # Update the state
        state = next_state
        
        total_reward += reward
        
        if done or truncated:
            break

    if episode % 100 == 0:
        print(f"Episode {episode}/{episodes} - Total Reward: {total_reward}")

env.close()

# Optionally, plot training progress (total reward over episodes)
# plt.plot(range(episodes), total_reward)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.show()
