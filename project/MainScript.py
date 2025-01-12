import numpy as np
import gymnasium as gym
# import wandb
from RewardParameterGenerator import RewardParameterGenerator
from StabilityChecker import StabilityChecker
from Runner import Runner
from agents.TD3_Agent import Agent
import my_gym_environments

# Initialize wandb
# wandb.login(key=["d351eeadb0e3c40b2b8a679d3a27c94f7fcd6f24"])
# wandb.init(project="rl-control-lti", entity="shivanichepuri")

# Step 1: Define the Linear Environment
env = gym.make("LinearFirstOrder-v0")
# try:
#     max_action = float(env.unwrapped.max_action)
# except AttributeError:
#     max_action = 1.0

# Step 2: Initialize the RL Agent
agent = Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    # max_action=max_action
)

# Step 3: Generate Reward Combinations
reward_gen = RewardParameterGenerator(
    w1_range=np.linspace(0.01, 10.0, 2),
    lam_range=np.linspace(0.01, 10.0, 2),
    lam2_range=np.linspace(0.01, 10.0, 2),
    w2_range=np.linspace(0.01, 10.0, 2)
)
reward_combinations = reward_gen.generate_combinations()
print("reward_combination : ", reward_combinations)

# Step 4: Filter Stable Reward Combinations
systemMatrixA = np.array([[-2.0]])
systemMatrixB = np.array([[3.0]])
checker = StabilityChecker(systemMatrixA, systemMatrixB)

stable_combinations = []
for w1, lam, lam2, w2 in reward_combinations:
    # Approximate gain derived from reward weights
    # G = w1 + lam + lam2
    G = w1 / (lam + lam2)
    stable, eigenvalues = checker.is_stable(G)
    if stable:
        stable_combinations.append((w1, lam, lam2, w2))

print("Stable Reward Combinations:", stable_combinations)

# Step 5: Train and Evaluate the RL Agent
runner = Runner(
    env=env,
    agent=agent,
    reward_combinations=stable_combinations,
    num_episodes=3,
    model_dir="models",  # Directory to save models
    results_csv="results.csv",  # CSV file to save performance metrics
    log_to_wandb=True  # Flag to log to wandb
)

for reward_params in stable_combinations:
    runner.train(reward_params)

runner.evaluate()