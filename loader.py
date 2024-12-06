import numpy as np

# Load the Q-table
with open("q_table_episode_100.npy", "rb") as f:
    Q = np.load(f, allow_pickle=True).item()

# Print the first few states and their action values
for state, action_values in list(Q.items()):  # Show 10 states for brevity
    print(f"State: {state}, Action Values: {action_values}")
