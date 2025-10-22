import gymnasium as gym
# Uncomment the next line if you install matplotlib: pip install matplotlib
# import matplotlib.pyplot as plt

def display_taxi_environment():
    # Create environment with rgb_array mode
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    observation, info = env.reset()
    
    print("Action Space:", env.action_space)
    print("State Space:", env.observation_space)
    print("\nActions:")
    print("0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff")
    
    # Take some actions and show frames
    for step in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Get the rendered frame
        frame = env.render()
        print(f"\nStep {step+1}:")
        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Frame shape: {frame.shape}")
        
        # If matplotlib is available, uncomment these lines:
        # plt.figure(figsize=(6, 6))
        # plt.imshow(frame)
        # plt.title(f"Step {step+1}: Action {action}, Reward {reward}")
        # plt.axis('off')
        # plt.show()
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    display_taxi_environment()