import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_q_value_change(csv_file):
    # Load metrics from CSV
    data = pd.read_csv(csv_file)

    # Compute change in mean Q-value
    data["Q-Value Change"] = data["Mean Q-Value"].diff()

    # Plot change in Q-values
    plt.figure(figsize=(10, 6))
    plt.plot(data["Episode"], data["Q-Value Change"], label="Change in Q-Value", color="red", marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Add horizontal line at 0
    plt.xlabel("Episode")
    plt.ylabel("Change in Mean Q-Value")
    plt.title("Change in Mean Q-Value Over Episodes")
    plt.grid()
    plt.legend()
    plt.show()

# Example Usage
if __name__ == "__main__":
    csv_file = "../metrics.csv"  # Path to your CSV file
    plot_q_value_change(csv_file)
