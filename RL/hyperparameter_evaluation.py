# Acknowledgement: This code contains code generated with the assistance of Claude Sonnet 4.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns

class QLearningHyperparameterEvaluator:
    def __init__(self):
        self.results = {}
        
    def train_agent(self, alpha, gamma, epsilon, episodes=1000, epsilon_decay=0.995, min_epsilon=0.01):
        """Train Q-learning agent with given hyperparameters"""
        env = gym.make("Taxi-v3")
        q_table = np.zeros((500, 6))
        
        rewards_per_episode = []
        steps_per_episode = []
        success_count = []
        current_epsilon = epsilon
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
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
                
                if steps > 200:  # Prevent infinite loops
                    break
            
            # Decay epsilon
            if current_epsilon > min_epsilon:
                current_epsilon *= epsilon_decay
            
            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
            success_count.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        # Calculate final metrics
        final_success_rate = np.mean(success_count[-100:])  # Last 100 episodes
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_steps = np.mean(steps_per_episode[-100:])
        
        return {
            'rewards': rewards_per_episode,
            'steps': steps_per_episode,
            'success_count': success_count,
            'final_success_rate': final_success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'q_table': q_table
        }
    
    def evaluate_alpha_values(self, alpha_values, gamma=0.6, epsilon=0.1, episodes=1000):
        """Evaluate different alpha values"""
        results = {}
        
        print("Evaluating Alpha values...")
        for alpha in alpha_values:
            print(f"Training with alpha = {alpha}")
            result = self.train_agent(alpha, gamma, epsilon, episodes)
            results[alpha] = result
            print(f"  Final Success Rate: {result['final_success_rate']:.3f}")
        
        return results
    
    def evaluate_gamma_values(self, gamma_values, alpha=0.1, epsilon=0.1, episodes=1000):
        """Evaluate different gamma values"""
        results = {}
        
        print("Evaluating Gamma values...")
        for gamma in gamma_values:
            print(f"Training with gamma = {gamma}")
            result = self.train_agent(alpha, gamma, epsilon, episodes)
            results[gamma] = result
            print(f"  Final Success Rate: {result['final_success_rate']:.3f}")
        
        return results
    
    def evaluate_epsilon_values(self, epsilon_values, alpha=0.1, gamma=0.6, episodes=1000):
        """Evaluate different epsilon values"""
        results = {}
        
        print("Evaluating Epsilon values...")
        for epsilon in epsilon_values:
            print(f"Training with epsilon = {epsilon}")
            result = self.train_agent(alpha, gamma, epsilon, episodes)
            results[epsilon] = result
            print(f"  Final Success Rate: {result['final_success_rate']:.3f}")
        
        return results
    
    def plot_hyperparameter_comparison(self, param_name, param_values, results, save_name):
        """Create comprehensive plots for hyperparameter comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Q-Learning Performance vs {param_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Learning Curves (Smoothed Rewards)
        ax1 = axes[0, 0]
        for param_val in param_values:
            rewards = results[param_val]['rewards']
            # Smooth using rolling average
            window = min(50, len(rewards)//10)
            smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            ax1.plot(smoothed, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Curves (Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Steps per Episode
        ax2 = axes[0, 1]
        for param_val in param_values:
            steps = results[param_val]['steps']
            window = min(50, len(steps)//10)
            smoothed_steps = [np.mean(steps[max(0, i-window):i+1]) for i in range(len(steps))]
            ax2.plot(smoothed_steps, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Average Steps')
        ax2.set_title('Learning Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate Over Time
        ax3 = axes[0, 2]
        for param_val in param_values:
            success = results[param_val]['success_count']
            # Calculate rolling success rate
            window = 100
            success_rate = [np.mean(success[max(0, i-window):i+1]) for i in range(window, len(success), 10)]
            episodes_x = range(window, len(success), 10)
            ax3.plot(episodes_x, success_rate, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Success Rate (100-episode window)')
        ax3.set_title('Success Rate Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final Performance Metrics
        ax4 = axes[1, 0]
        final_rewards = [results[param_val]['avg_reward'] for param_val in param_values]
        bars = ax4.bar(range(len(param_values)), final_rewards, alpha=0.7, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(param_values))))
        ax4.set_xlabel(f'{param_name.upper()} Values')
        ax4.set_ylabel('Final Average Reward')
        ax4.set_title('Final Performance: Average Reward')
        ax4.set_xticks(range(len(param_values)))
        ax4.set_xticklabels(param_values)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 5: Final Success Rates
        ax5 = axes[1, 1]
        final_success = [results[param_val]['final_success_rate'] for param_val in param_values]
        bars = ax5.bar(range(len(param_values)), final_success, alpha=0.7, 
                      color=plt.cm.plasma(np.linspace(0, 1, len(param_values))))
        ax5.set_xlabel(f'{param_name.upper()} Values')
        ax5.set_ylabel('Final Success Rate')
        ax5.set_title('Final Performance: Success Rate')
        ax5.set_xticks(range(len(param_values)))
        ax5.set_xticklabels(param_values)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 6: Average Steps to Solution
        ax6 = axes[1, 2]
        avg_steps = [results[param_val]['avg_steps'] for param_val in param_values]
        bars = ax6.bar(range(len(param_values)), avg_steps, alpha=0.7, 
                      color=plt.cm.coolwarm(np.linspace(0, 1, len(param_values))))
        ax6.set_xlabel(f'{param_name.upper()} Values')
        ax6.set_ylabel('Average Steps to Solution')
        ax6.set_title('Efficiency: Steps per Episode')
        ax6.set_xticks(range(len(param_values)))
        ax6.set_xticklabels(param_values)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_heatmap_analysis(self, alpha_values, gamma_values, epsilon_values):
        """Create heatmap analysis for combinations of hyperparameters"""
        print("Creating heatmap analysis...")
        
        # Test combinations of alpha and gamma
        alpha_gamma_results = np.zeros((len(alpha_values), len(gamma_values)))
        
        for i, alpha in enumerate(alpha_values):
            for j, gamma in enumerate(gamma_values):
                print(f"Testing alpha={alpha}, gamma={gamma}")
                result = self.train_agent(alpha, gamma, epsilon=0.1, episodes=500)
                alpha_gamma_results[i, j] = result['final_success_rate']
        
        # Test combinations of alpha and epsilon
        alpha_epsilon_results = np.zeros((len(alpha_values), len(epsilon_values)))
        
        for i, alpha in enumerate(alpha_values):
            for j, epsilon in enumerate(epsilon_values):
                print(f"Testing alpha={alpha}, epsilon={epsilon}")
                result = self.train_agent(alpha, gamma=0.6, epsilon=epsilon, episodes=500)
                alpha_epsilon_results[i, j] = result['final_success_rate']
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Alpha vs Gamma heatmap
        sns.heatmap(alpha_gamma_results, 
                   xticklabels=gamma_values, 
                   yticklabels=alpha_values,
                   annot=True, fmt='.3f', cmap='viridis',
                   ax=axes[0])
        axes[0].set_title('Success Rate: Alpha vs Gamma')
        axes[0].set_xlabel('Gamma (Discount Factor)')
        axes[0].set_ylabel('Alpha (Learning Rate)')
        
        # Alpha vs Epsilon heatmap
        sns.heatmap(alpha_epsilon_results,
                   xticklabels=epsilon_values,
                   yticklabels=alpha_values,
                   annot=True, fmt='.3f', cmap='plasma',
                   ax=axes[1])
        axes[1].set_title('Success Rate: Alpha vs Epsilon')
        axes[1].set_xlabel('Epsilon (Exploration Rate)')
        axes[1].set_ylabel('Alpha (Learning Rate)')
        
        plt.tight_layout()
        plt.savefig('hyperparameter_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    evaluator = QLearningHyperparameterEvaluator()
    
    # Define hyperparameter ranges
    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_values = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99]
    epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    
    episodes = 1000
    
    # Evaluate each hyperparameter individually
    print("="*60)
    print("HYPERPARAMETER EVALUATION FOR Q-LEARNING")
    print("="*60)
    
    # 1. Evaluate Alpha values
    alpha_results = evaluator.evaluate_alpha_values(alpha_values, episodes=episodes)
    evaluator.plot_hyperparameter_comparison('alpha', alpha_values, alpha_results, 'alpha_comparison')
    
    # 2. Evaluate Gamma values  
    gamma_results = evaluator.evaluate_gamma_values(gamma_values, episodes=episodes)
    evaluator.plot_hyperparameter_comparison('gamma', gamma_values, gamma_results, 'gamma_comparison')
    
    # 3. Evaluate Epsilon values
    epsilon_results = evaluator.evaluate_epsilon_values(epsilon_values, episodes=episodes)
    evaluator.plot_hyperparameter_comparison('epsilon', epsilon_values, epsilon_results, 'epsilon_comparison')
    
    # 4. Create heatmap analysis (reduced values for computational efficiency)
    alpha_heatmap = [0.05, 0.1, 0.3, 0.5, 0.7]
    gamma_heatmap = [0.3, 0.6, 0.8, 0.9, 0.99]
    epsilon_heatmap = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    evaluator.create_heatmap_analysis(alpha_heatmap, gamma_heatmap, epsilon_heatmap)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF BEST PERFORMING HYPERPARAMETERS")
    print("="*60)
    
    # Find best alpha
    best_alpha = max(alpha_results.keys(), key=lambda k: alpha_results[k]['final_success_rate'])
    print(f"Best Alpha: {best_alpha} (Success Rate: {alpha_results[best_alpha]['final_success_rate']:.3f})")
    
    # Find best gamma
    best_gamma = max(gamma_results.keys(), key=lambda k: gamma_results[k]['final_success_rate'])
    print(f"Best Gamma: {best_gamma} (Success Rate: {gamma_results[best_gamma]['final_success_rate']:.3f})")
    
    # Find best epsilon
    best_epsilon = max(epsilon_results.keys(), key=lambda k: epsilon_results[k]['final_success_rate'])
    print(f"Best Epsilon: {best_epsilon} (Success Rate: {epsilon_results[best_epsilon]['final_success_rate']:.3f})")
    
    print(f"\nRecommended hyperparameters: alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon}")

if __name__ == "__main__":
    main()