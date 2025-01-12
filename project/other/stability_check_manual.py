import numpy as np
from RewardParameterGenerator import RewardParameterGenerator
from StabilityChecker import StabilityChecker
from PerformanceMetricsCalculator import PerformanceMetricsCalculator
from Runner import Runner
from agents.TD3_Agent import Agent
import gymnasium as gym
import my_gym_environments  # Ensure the custom environment is registered

# Step 1: Define the Linear Environment
env = gym.make("LinearFirstOrder-v0")  # Replace with the actual registered ID of the environment

# Step 2: Initialize the RL Agent
agent = Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_action=float(env.max_action)
)

# Step 3: Generate Reward Combinations
reward_gen = RewardParameterGenerator(
    w1_range=np.linspace(0.1, 2.0, 5),  # Example range for w1
    lam_range=np.linspace(0.1, 2.0, 5),  # Example range for lam
    lam2_range=np.linspace(0.1, 2.0, 5),  # Example range for lam2
    w2_range=np.linspace(0.01, 0.1, 3)  # Example range for w2
)
reward_combinations = reward_gen.generate_combinations()

# Step 4: Filter Stable Reward Combinations
A = np.array([[-2.0]])  # System A matrix
B = np.array([[3.0]])   # System B matrix
checker = StabilityChecker(A, B)

stable_combinations = []
for w1, lam, lam2, w2 in reward_combinations:
    G = w1 + lam + lam2  # Approximate gain derived from reward weights
    stable, eigenvalues = checker.is_stable(G)
    if stable:
        stable_combinations.append((w1, lam, lam2, w2))

print("Stable Reward Combinations:", stable_combinations)

# Step 5: Train and Evaluate the RL Agent
runner = Runner(
    env=env,
    agent=agent,
    reward_combinations=stable_combinations,
    num_episodes=150,
    model_dir="models",  # Directory to save models
    results_csv="results.csv"  # CSV file to save performance metrics
)

# Train the RL agent
runner.train()

# Evaluate the RL agent
runner.evaluate()
