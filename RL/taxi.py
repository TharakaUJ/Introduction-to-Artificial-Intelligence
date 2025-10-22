import gymnasium as gym
import numpy as np


q_table = np.zeros((500, 6))
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate


env = gym.make("Taxi-v3", render_mode="human")
env = gym.make("Taxi-v3")

for episodes in range(10):
    done = False
    i = 0

    previous_observation = env.reset()[0]
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}")
        i += 1

        q_table[previous_observation, action] = q_table[previous_observation, action] * (1-alpha) + alpha * (reward + gamma * np.max(q_table[observation]))
        previous_observation = observation

        done = terminated or truncated
        
        if terminated or truncated:
            previous_observation = env.reset()[0]

    env.reset()

env.close()


