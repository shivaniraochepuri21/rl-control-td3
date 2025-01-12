import pandas as pd
import matplotlib.pyplot as plt

def plot_average_reward_vs_episodes(results_csv):
    # Load the CSV file
    df = pd.read_csv(results_csv)

    # Extract the average reward per episode (use Total Reward as proxy for now)
    avg_rewards = df["Total Reward"]

    # Plot average rewards vs episodes
    plt.plot(avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Average Reward vs Episodes")
    plt.grid(True)
    plt.show()

# Example usage:
plot_average_reward_vs_episodes("results.csv")
