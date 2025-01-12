# import gymnasium as gym
# import my_gym_environments
#
# env = gym.make('LinearFirstOrder-v0')
#
# obs = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()
#     print("env.step(action)", env.step(action))
#     obs, reward, done, info, done_ = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
#
# env.close()
#

import gymnasium as gym
import my_gym_environments  # Assuming your custom environments are defined here

# Create environment
env = gym.make('LinearFirstOrder-v0')

# Reset the environment (adjust for the version of Gymnasium you're using)
obs, info = env.reset()  # If reset() returns two values: (obs, info)

# Run the environment for 10 steps
for _ in range(10):
    # Sample an action
    action = env.action_space.sample()

    # Take a step in the environment and unpack the result
    obs, reward, done, truncated, info = env.step(action)  # Updated to 5-tuple
    print(f"env.step(action): {obs}, reward: {reward}, done: {done}, truncated: {truncated}, info: {info}")

    # Render the environment (only if implemented)
    try:
        env.render()
    except NotImplementedError:
        print("Rendering not implemented.")

    # Reset environment if done
    if done or truncated:
        obs, info = env.reset()

# Close the environment
env.close()
