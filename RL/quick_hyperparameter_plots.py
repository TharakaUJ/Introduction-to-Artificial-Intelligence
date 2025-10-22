# Acknowledgement: This code contains code generated with the assistance of Claude Sonnet 4.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def quick_train_agent(alpha, gamma, epsilon, episodes=500):
    """Quickly train an agent with given hyperparameters"""
    env = gym.make("Taxi-v3")
    q_table = np.zeros((500, 6))
    
    rewards = []
    success_rates = []
    current_epsilon = epsilon
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # Epsilon-greedy action selection
            if np.random.random() < current_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
            )
            
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Decay epsilon
        if current_epsilon > 0.01:
            current_epsilon *= 0.995
        
        rewards.append(total_reward)
        
        # Calculate success rate for last 50 episodes
        if episode >= 49:
            recent_success = sum(1 for r in rewards[-50:] if r > 0) / 50
            success_rates.append(recent_success)
    
    env.close()
    
    final_success_rate = np.mean([1 if r > 0 else 0 for r in rewards[-100:]])
    avg_reward = np.mean(rewards[-100:])
    
    return {
        'rewards': rewards,
        'success_rates': success_rates,
        'final_success_rate': final_success_rate,
        'avg_reward': avg_reward
    }

def plot_hyperparameter_effects():
    """Create focused plots showing hyperparameter effects"""
    
    # Test different values
    alpha_values = [0.01, 0.1, 0.3, 0.5, 0.8]
    gamma_values = [0.1, 0.5, 0.8, 0.95, 0.99]
    epsilon_values = [0.01, 0.1, 0.2, 0.4, 0.6]
    
    episodes = 500
    
    print("Testing Alpha values...")
    alpha_results = {}
    for alpha in alpha_values:
        print(f"  α = {alpha}")
        alpha_results[alpha] = quick_train_agent(alpha, 0.6, 0.1, episodes)
    
    print("Testing Gamma values...")
    gamma_results = {}
    for gamma in gamma_values:
        print(f"  γ = {gamma}")
        gamma_results[gamma] = quick_train_agent(0.1, gamma, 0.1, episodes)
    
    print("Testing Epsilon values...")
    epsilon_results = {}
    for epsilon in epsilon_values:
        print(f"  ε = {epsilon}")
        epsilon_results[epsilon] = quick_train_agent(0.1, 0.6, epsilon, episodes)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Alpha Analysis
    plt.subplot(3, 3, 1)
    for alpha in alpha_values:
        rewards = alpha_results[alpha]['rewards']
        smoothed = [np.mean(rewards[max(0, i-25):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed, label=f'α={alpha}', linewidth=2)
    plt.title('Alpha Effect on Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    success_rates = [alpha_results[alpha]['final_success_rate'] for alpha in alpha_values]
    bars = plt.bar(range(len(alpha_values)), success_rates, alpha=0.7, color='skyblue')
    plt.title('Final Success Rate vs Alpha', fontsize=14, fontweight='bold')
    plt.xlabel('Alpha (Learning Rate)')
    plt.ylabel('Success Rate')
    plt.xticks(range(len(alpha_values)), alpha_values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 3)
    for alpha in alpha_values:
        success_rates = alpha_results[alpha]['success_rates']
        episodes_x = range(50, len(success_rates) + 50)
        plt.plot(episodes_x, success_rates, label=f'α={alpha}', linewidth=2)
    plt.title('Success Rate Evolution (Alpha)', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (50-ep window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gamma Analysis
    plt.subplot(3, 3, 4)
    for gamma in gamma_values:
        rewards = gamma_results[gamma]['rewards']
        smoothed = [np.mean(rewards[max(0, i-25):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed, label=f'γ={gamma}', linewidth=2)
    plt.title('Gamma Effect on Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 5)
    success_rates = [gamma_results[gamma]['final_success_rate'] for gamma in gamma_values]
    bars = plt.bar(range(len(gamma_values)), success_rates, alpha=0.7, color='lightgreen')
    plt.title('Final Success Rate vs Gamma', fontsize=14, fontweight='bold')
    plt.xlabel('Gamma (Discount Factor)')
    plt.ylabel('Success Rate')
    plt.xticks(range(len(gamma_values)), gamma_values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    for gamma in gamma_values:
        success_rates = gamma_results[gamma]['success_rates']
        episodes_x = range(50, len(success_rates) + 50)
        plt.plot(episodes_x, success_rates, label=f'γ={gamma}', linewidth=2)
    plt.title('Success Rate Evolution (Gamma)', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (50-ep window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Epsilon Analysis
    plt.subplot(3, 3, 7)
    for epsilon in epsilon_values:
        rewards = epsilon_results[epsilon]['rewards']
        smoothed = [np.mean(rewards[max(0, i-25):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed, label=f'ε={epsilon}', linewidth=2)
    plt.title('Epsilon Effect on Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    success_rates = [epsilon_results[epsilon]['final_success_rate'] for epsilon in epsilon_values]
    bars = plt.bar(range(len(epsilon_values)), success_rates, alpha=0.7, color='salmon')
    plt.title('Final Success Rate vs Epsilon', fontsize=14, fontweight='bold')
    plt.xlabel('Epsilon (Exploration Rate)')
    plt.ylabel('Success Rate')
    plt.xticks(range(len(epsilon_values)), epsilon_values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 9)
    for epsilon in epsilon_values:
        success_rates = epsilon_results[epsilon]['success_rates']
        episodes_x = range(50, len(success_rates) + 50)
        plt.plot(episodes_x, success_rates, label=f'ε={epsilon}', linewidth=2)
    plt.title('Success Rate Evolution (Epsilon)', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (50-ep window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_effects_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK HYPERPARAMETER ANALYSIS RESULTS")
    print("="*60)
    
    best_alpha = max(alpha_results.keys(), key=lambda k: alpha_results[k]['final_success_rate'])
    best_gamma = max(gamma_results.keys(), key=lambda k: gamma_results[k]['final_success_rate'])
    best_epsilon = max(epsilon_results.keys(), key=lambda k: epsilon_results[k]['final_success_rate'])
    
    print(f"Best Alpha:   {best_alpha:5.2f} (Success: {alpha_results[best_alpha]['final_success_rate']:.3f})")
    print(f"Best Gamma:   {best_gamma:5.2f} (Success: {gamma_results[best_gamma]['final_success_rate']:.3f})")
    print(f"Best Epsilon: {best_epsilon:5.2f} (Success: {epsilon_results[best_epsilon]['final_success_rate']:.3f})")
    
    print(f"\nRecommended: α={best_alpha}, γ={best_gamma}, ε={best_epsilon}")
    
    # Create comparison table
    print(f"\nDetailed Results:")
    print(f"{'Parameter':<10} {'Value':<8} {'Success Rate':<12} {'Avg Reward':<10}")
    print("-" * 45)
    
    for alpha in alpha_values:
        result = alpha_results[alpha]
        print(f"{'Alpha':<10} {alpha:<8.2f} {result['final_success_rate']:<12.3f} {result['avg_reward']:<10.2f}")
    
    print()
    for gamma in gamma_values:
        result = gamma_results[gamma]
        print(f"{'Gamma':<10} {gamma:<8.2f} {result['final_success_rate']:<12.3f} {result['avg_reward']:<10.2f}")
    
    print()
    for epsilon in epsilon_values:
        result = epsilon_results[epsilon]
        print(f"{'Epsilon':<10} {epsilon:<8.2f} {result['final_success_rate']:<12.3f} {result['avg_reward']:<10.2f}")

if __name__ == "__main__":
    plot_hyperparameter_effects()