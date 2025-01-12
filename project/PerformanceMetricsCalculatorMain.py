import numpy as np
from PerformanceMetricsCalculator import PerformanceMetricsCalculator

# Sample data
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])                      # Time vector (seconds)
y = np.array([0, 2, 5, 7, 9.0, 10.0, 10.0, 10.0, 10.0])      # System response
u = np.array([0, 2, -1, 3, 0, 0, 0, 0, 0])                    # Control input
r = 10.0                                                      # Desired final output (target state)

# Initialize the PerformanceMetricsCalculator with the target_state
calculator = PerformanceMetricsCalculator(target_state=r)

# 1. Settling Time
Ts = calculator.calculate_settling_time(t, y)
if Ts is not None:
    print(f"Settling Time: {Ts} seconds")
else:
    print("Settling Time: Not found within the data range.")

# 2. Rise Time
Tr = calculate_rise_time(t, y)
if Tr is not None:
    print(f"Rise Time: {Tr} seconds")
else:
    print("Rise Time: Not found within the data range.")

# 3. Control Effort
# a. Integral of |u(t)|
control_effort_abs = calculator.calculate_control_effort(t, u, squared=False, difference=False)
print(f"Control Effort (Integral of |u(t)|): {control_effort_abs} units")

# b. Integral of u(t)^2
control_effort_squared = calculator.calculate_control_effort(t, u, squared=True, difference=False)
print(f"Control Effort (Integral of u(t)^2): {control_effort_squared} units^2")

# c. Integral of |u[i] - u[i-1]| (Difference Absolute)
control_effort_diff_abs = calculator.calculate_control_effort(t, u, squared=False, difference=True)
print(f"Control Effort (Integral of |Δu(t)|): {control_effort_diff_abs} units")

# d. Integral of (u[i] - u[i-1])^2 (Difference Squared)
control_effort_diff_squared = calculator.calculate_control_effort(t, u, squared=True, difference=True)
print(f"Control Effort (Integral of Δu(t)^2): {control_effort_diff_squared} units^2")

# 4. Steady-State Error
sse = calculator.calculate_steady_state_error(y[-1])  # Use the last value in the response
print(f"Steady-State Error: {sse} units")

# 5. Steady-State Error Percentage
sse_percent = calculator.calculate_steady_state_error_percentage(y[-1])  # Use the last value in the response
print(f"Steady-State Error Percentage: {sse_percent}%")

# 6. Overshoot
Mp = calculate_overshoot(y)
print(f"Overshoot: {Mp} units")

# 7. Overshoot Percentage
Mp_percent = calculator.calculate_overshoot_percentage(y)
print(f"Overshoot Percentage: {Mp_percent}%")

# Optional: Calculate Comprehensive Metrics from Episode Data
# Define episode_data as a list of dictionaries with keys 'state', 'action', 'error', 'reward', 'time_step'
episode_data = [
    {'state': 0, 'action': 0, 'error': 10, 'reward': 0, 'time_step': 0},
    {'state': 2, 'action': 2, 'error': 8, 'reward': 1, 'time_step': 1},
    {'state': 5, 'action': -1, 'error': 5, 'reward': 2, 'time_step': 2},
    {'state': 7, 'action': 3, 'error': 3, 'reward': 3, 'time_step': 3},
    {'state': 9, 'action': 0, 'error': 1, 'reward': 4, 'time_step': 4},
    {'state': 10, 'action': 0, 'error': 0, 'reward': 5, 'time_step': 5},
    {'state': 10, 'action': 0, 'error': 0, 'reward': 6, 'time_step': 6},
    {'state': 10, 'action': 0, 'error': 0, 'reward': 7, 'time_step': 7},
    {'state': 10, 'action': 0, 'error': 0, 'reward': 8, 'time_step': 8}
]

metrics = calculator.calculate_metrics(episode_data)
print("Comprehensive Metrics:", metrics)
