import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_max_q_values(pickle_dir):
    pickle_files = sorted([f for f in os.listdir(pickle_dir) if f.startswith("q_table_episode_") and f.endswith(".pkl")])
    q_values_max = []

    for pickle_file in pickle_files:
        pickle_path = os.path.join(pickle_dir, pickle_file)
        with open(pickle_path, "rb") as f:
            q_table = pickle.load(f)
            q_values = []
            for v in q_table.values():
                v = np.array(v)
                q_values.append(v.flatten() if v.ndim > 1 else v)
            q_values = np.concatenate(q_values)
            q_values_max.append(np.max(q_values))

    plt.plot(range(len(q_values_max)), q_values_max, marker="o", label="Max Q-Value")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-Value")
    plt.title("Max Q-Value Trends Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()

# Example Usage
if __name__ == "__main__":
    pickle_dir = ".."
    plot_max_q_values(pickle_dir)
