import numpy as np

class PerformanceMetricsCalculator:
    def __init__(self, target_state=0.0, settling_threshold=0.02):
        self.target_state = target_state
        self.settling_threshold = settling_threshold

    def calculate_metrics(self, episode_data):
        """
        Calculate performance metrics based on episode data.
        
        Args:
            episode_data (list of dict): A list where each element is a dictionary
                containing 'state', 'action', 'error', 'reward', 'time_step'.
        
        Returns:
            dict: A dictionary containing calculated performance metrics.
        """
        # Extract data
        states = np.array([step['state'] for step in episode_data])
        actions = np.array([step['action'] for step in episode_data])
        rewards = np.array([step['reward'] for step in episode_data])
        errors = np.array([step['error'] for step in episode_data])
        time_steps = np.arange(len(episode_data))

        # Calculate Settling Time
        settling_time = None
        for i, state in enumerate(states):
            if np.abs(state - self.target_state) < self.settling_threshold:
                settling_time = i
                break

        # Calculate Rise Time
        rise_time = None
        for i, state in enumerate(states):
            if state >= 0.9 * self.target_state:
                rise_time = i
                break

        # Calculate Overshoot
        overshoot = np.max(states) - self.target_state

        # Calculate Control Effort
        control_effort = np.sum(np.abs(actions))

        # Calculate Total Reward
        total_reward = np.sum(rewards)

        # Calculate Steady-State Error (SSE)
        steady_state_error = np.abs(states[-1] - self.target_state)

        return {
            "Settling Time": settling_time,
            "Rise Time": rise_time,
            "Overshoot": overshoot,
            "Control Effort": control_effort,
            "Total Reward": total_reward,
            "Steady-State Error": steady_state_error,
        }
