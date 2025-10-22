import gymnasium as gym
import numpy as np

# Initialize Q-table and hyperparameters
q_table = np.zeros((500, 6))
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

env = gym.make("Taxi-v3", render_mode="human")

def choose_action(state, epsilon):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

for episode in range(100):
    state, _ = env.reset()  
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = choose_action(state, epsilon)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
        )
        
        state = next_state
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps > 200:
            break
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}")

env.close()

print("\nTesting trained agent...")
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Test episode reward: {total_reward}")
env.close()


