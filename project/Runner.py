import csv
import os
import wandb
from project.PerformanceMetricsCalculator import PerformanceMetricsCalculator
import numpy as np

class Runner:
    def __init__(self, env, agent, reward_combinations, num_episodes=10, model_dir="models", results_csv="results.csv",
                 log_to_wandb=False):
        self.env = env
        self.agent = agent
        self.reward_combinations = reward_combinations
        self.num_episodes = num_episodes
        self.model_dir = model_dir
        self.results_csv = results_csv
        self.log_to_wandb = log_to_wandb
        self.metrics_calculator = PerformanceMetricsCalculator(target_state=self.env.targ_state)

        # Initialize wandb if logging is enabled
        if self.log_to_wandb:
            wandb.login(key=["d351eeadb0e3c40b2b8a679d3a27c94f7fcd6f24"])
            wandb.init(project="rl-control-lti", entity="shivanichepuri")
            wandb.config.update({
                "num_episodes": num_episodes,
                "model_dir": model_dir,
                "results_csv": results_csv
            })

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Prepare CSV for results
        with open(self.results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "w1", "lam", "lam2", "w2",
                "Settling Time", "Rise Time", "Overshoot",
                "Control Effort", "Total Reward", "Steady-State Error"
            ])

    def log_to_wandb_metrics(self, episode_data, episode):

        """ Log relevant episode data to wandb """
        rewards = [step["reward"] for step in episode_data]
        avg_reward = np.mean(rewards)
        total_reward = np.sum(rewards)

        # Log to wandb
        if self.log_to_wandb:
            wandb.log({"Episode": episode, "Average Reward": avg_reward})
            wandb.log({"Episode": episode, "Total Reward": total_reward})

    def train(self, reward_params):
        w1, lam, lam2, w2 = reward_params
        print(f"Training with rewards: w1={w1}, lam={lam}, lam2={lam2}, w2={w2}")

        # Update environment with current reward parameters
        self.env.w1, self.env.lam, self.env.lam2, self.env.w2 = w1, lam, lam2, w2

        episode_data = []  # To store episode details for metrics
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _, info = self.env.step(action)

                # Record step data
                episode_data.append({
                    "state": state[0],
                    "action": action[0],
                    "reward": reward,
                    "info": info,
                    "time_step": len(episode_data)
                })

                self.agent.memorize(state, action, reward, next_state, done)
                err_actor, err_critic = self.agent.learn()
                state = next_state

            # Log to wandb after each episode
            self.log_to_wandb_metrics(episode_data, episode)

        # Calculate performance metrics for the current parameter combination
        metrics = self.metrics_calculator.calculate_metrics(episode_data)

        # Save performance metrics
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([w1, lam, lam2, w2] + list(metrics.values()))

        # Optionally save trained model
        model_name = f"{self.model_dir}/model_w1_{w1}_lam_{lam}_lam2_{lam2}_w2_{w2}.pt"
        self.agent.TD.save(model_name)

    def evaluate(self):
        state, _ = self.env.reset()
        done = False
        episode_data = []
        while not done:
            action = self.agent.act(state)
            next_state, reward, done, _, info = self.env.step(action)

            # Record step data
            episode_data.append({
                "state": state[0],
                "action": action[0],
                "reward": reward,
                "error": info["error"],
                "time_step": len(episode_data)
            })
            state = next_state

        # Calculate and print evaluation metrics
        metrics = self.metrics_calculator.calculate_metrics(episode_data)
        print(f"Evaluation Metrics: {metrics}")

