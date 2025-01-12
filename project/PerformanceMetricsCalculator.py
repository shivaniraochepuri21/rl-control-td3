import numpy as np

class PerformanceMetricsCalculator:
    def __init__(self, target_state=0.0, settling_threshold=0.02):
        self.target_state = target_state
        self.settling_threshold = settling_threshold

    def calculate_overshoot(self, y, y_final):
        """
        Calculate the absolute overshoot.

        Parameters:
        - y: Response vector (1D array).
        - y_final: Actual final output (float).

        Returns:
        - Overshoot (float).
        """
        y_max = np.max(y)
        return y_max - y_final

    def calculate_rise_time(self, t, y, lower_percent=0.1, upper_percent=0.9):
        """
        Calculate the rise time of the system response.

        Parameters:
        - t: Time vector (1D array).
        - y: Response vector (1D array).
        - lower_percent: Lower percentage threshold (default 10%).
        - upper_percent: Upper percentage threshold (default 90%).

        Returns:
        - Rise time (float) in the same units as t, or None if not found.
        """
        y_final = y[-1]
        y_lower = y_final * lower_percent
        y_upper = y_final * upper_percent

        # Find t_lower
        t_lower = None
        for i in range(len(t)):
            if y[i] >= y_lower:
                t_lower = t[i]
                break

        # Find t_upper
        t_upper = None
        for i in range(len(t)):
            if t[i] >= t_lower and y[i] >= y_upper:
                t_upper = t[i]
                break

        if t_lower is not None and t_upper is not None:
            return t_upper - t_lower
        else:
            return None  # If rise time not found

    def calculate_settling_time(self, t, y, tolerance=None):
        """
        Calculate the settling time of the system response.

        Parameters:
        - t: Time vector (1D array).
        - y: Response vector (1D array).
        - tolerance: Tolerance band (default 2% if None).

        Returns:
        - Settling time (float) in the same units as t, or None if not found.
        """
        if tolerance is None:
            tolerance = self.settling_threshold  # Use the default threshold if not specified

        y_final = y[-1]
        upper_bound = y_final * (1 + tolerance)
        lower_bound = y_final * (1 - tolerance)

        for i in range(len(t)):
            # Check if all subsequent y are within bounds
            if np.all((y[i:] >= lower_bound) & (y[i:] <= upper_bound)):
                return t[i]
        return None  # If settling time not found

    def calculate_control_effort(self, t, u, squared=False, difference=False):
        """
        Calculate the control effort based on specified flags.

        Parameters:
        - t: Time vector (1D array).
        - u: Control input vector (1D array).
        - squared: If True, squares the control input.
        - difference: If True, squares the difference between consecutive control inputs.

        Returns:
        - Control effort (float).
        """
        effort = 0
        for i in range(1, len(t)):
            delta_t = t[i] - t[i - 1]
            if difference:
                delta_u = u[i] - u[i - 1]
                if squared:
                    term = (delta_u) ** 2
                else:
                    term = abs(delta_u)
            else:
                if squared:
                    term = u[i] ** 2
                else:
                    term = abs(u[i])

            # Trapezoidal integration
            effort += term * delta_t

        return effort

    def calculate_steady_state_error(self, r, y_final):
        """
        Calculate the absolute steady-state error.

        Parameters:
        - r: Desired final output (float).
        - y_final: Actual final output (float).

        Returns:
        - Steady-state error (float).
        """
        return r - y_final

    def calculate_steady_state_error_percentage(self, r, y_final):
        """
        Calculate the steady-state error as a percentage.

        Parameters:
        - r: Desired final output (float).
        - y_final: Actual final output (float).

        Returns:
        - Steady-state error percentage (float).
        """
        if r == 0:
            return np.nan  # Avoid division by zero
        return (self.calculate_steady_state_error(r, y_final) / abs(r)) * 100

    def calculate_overshoot_percentage(self, y, y_final):
        """
        Calculate the overshoot as a percentage.

        Parameters:
        - y: Response vector (1D array).
        - y_final: Actual final output (float).

        Returns:
        - Overshoot percentage (float).
        """
        if y_final == 0:
            return np.nan  # Avoid division by zero
        return (self.calculate_overshoot(y, y_final) / abs(y_final)) * 100

    def calculate_metrics(self, episode_data):
        """
        Calculate performance metrics based on episode data.

        Args:
            episode_data (list of dict): A list where each element is a dictionary
                containing 'state', 'action', 'error', 'reward', 'time_step'.

        Returns:
            dict: A dictionary containing calculated performance metrics.
        """
        # Extract data from episode_data
        states = np.array([step['state'] for step in episode_data])
        actions = np.array([step['action'] for step in episode_data])
        rewards = np.array([step['reward'] for step in episode_data])
        errors = np.array([step['error'] for step in episode_data])
        time_steps = np.array([step['time_step'] for step in episode_data])

        # Calculate Settling Time
        Ts = self.calculate_settling_time(time_steps, states)

        # Calculate Rise Time
        Tr = self.calculate_rise_time(time_steps, states)

        # Calculate Overshoot
        Mp = self.calculate_overshoot(states, states[-1])

        # Calculate Overshoot Percentage
        Op = self.calculate_overshoot_percentage(states, states[-1])

        # Calculate Control Effort (Integral of |action|)
        control_effort = self.calculate_control_effort(time_steps, actions, squared=False, difference=False)

        # Calculate return of the Episode
        episode_return = np.sum(rewards)

        # Calculate SSE
        sse = self.calculate_steady_state_error(states[-1], states[-1])

        # Calculate Steady-State Error Percentage
        sse_percent = self.calculate_steady_state_error_percentage(states[-1], states[-1])

        return {
            "Settling Time": Ts,
            "Rise Time": Tr,
            "Overshoot": Mp,
            "Overshoot Percentage" : Op,
            "Control Effort": control_effort,
            "Total Episode Reward": episode_return,
            "Steady-State Error": sse,
            "Steady-State Error Percentage": sse_percent,
        }
