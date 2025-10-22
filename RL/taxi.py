import gymnasium as gym
import time


env = gym.make("Taxi-v3", render_mode="human")
env.reset()

for episodes in range(10):
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}")
        i += 1
        done = terminated or truncated
        # Add a small delay to see the animation
        time.sleep(0.5)
        
        if terminated or truncated:
            env.reset()

    env.reset()

env.close()
