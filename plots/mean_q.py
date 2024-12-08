import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Plot Mean Q-Value Trends
def plot_mean_q_values(pickle_dir):
    """
    Load pickle files from a directory, compute mean Q-values,
    and plot the trend across episodes.

    Args:
    - pickle_dir: The directory containing pickle files.

    Output:
    - A plot of mean Q-value trends.
    """
    # Find pickle files
    pickle_files = sorted([f for f in os.listdir(pickle_dir) 
                           if f.startswith("q_table_episode_") and f.endswith(".pkl")])

    q_values_mean = []

    # Process each file
    for pickle_file in pickle_files:
        pickle_path = os.path.join(pickle_dir, pickle_file)
        with open(pickle_path, "rb") as f:
            q_table = pickle.load(f)
            
            # Flatten all Q-values into 1D arrays and concatenate them
            q_values = []
            for v in q_table.values():
                # Ensure all values are numpy arrays
                v = np.array(v)
                # Flatten if necessary
                q_values.append(v.flatten() if v.ndim > 1 else v)
            
            # Concatenate and compute the mean
            q_values = np.concatenate(q_values)
            q_values_mean.append(np.mean(q_values))

    # Plot the results
    plt.plot(range(len(q_values_mean)), q_values_mean, marker="o", label="Mean Q-Value")
    plt.xlabel("Episode")
    plt.ylabel("Mean Q-Value")
    plt.title("Mean Q-Value Trends Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
pickle_dir = ".."  # Parent directory containing pickle files
plot_mean_q_values(pickle_dir)
