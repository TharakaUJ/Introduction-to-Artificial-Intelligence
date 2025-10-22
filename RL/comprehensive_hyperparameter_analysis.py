# Acknowledgement: This code contains code generated with the assistance of Claude Sonnet 4.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningHyperparameterEvaluator:
    def __init__(self):
        self.results = {}
        
    def train_agent(self, alpha, gamma, epsilon, episodes=800, epsilon_decay=0.995, min_epsilon=0.01):
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
            'avg_steps': avg_steps
        }
    
    def plot_hyperparameter_comparison(self, param_name, param_values, results, save_name):
        """Create comprehensive plots for hyperparameter comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Q-Learning Performance vs {param_name.upper()}', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
        
        # Plot 1: Learning Curves (Smoothed Rewards)
        ax1 = axes[0, 0]
        for i, param_val in enumerate(param_values):
            rewards = results[param_val]['rewards']
            # Smooth using rolling average
            window = min(50, len(rewards)//10)
            smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            ax1.plot(smoothed, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2, color=colors[i])
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Curves (Smoothed)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Steps per Episode
        ax2 = axes[0, 1]
        for i, param_val in enumerate(param_values):
            steps = results[param_val]['steps']
            window = min(50, len(steps)//10)
            smoothed_steps = [np.mean(steps[max(0, i-window):i+1]) for i in range(len(steps))]
            ax2.plot(smoothed_steps, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2, color=colors[i])
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Average Steps')
        ax2.set_title('Learning Efficiency')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate Over Time
        ax3 = axes[0, 2]
        for i, param_val in enumerate(param_values):
            success = results[param_val]['success_count']
            # Calculate rolling success rate
            window = 50
            success_rate = []
            episodes_x = []
            for j in range(window, len(success), 10):
                success_rate.append(np.mean(success[max(0, j-window):j+1]))
                episodes_x.append(j)
            ax3.plot(episodes_x, success_rate, label=f'{param_name}={param_val}', alpha=0.8, linewidth=2, color=colors[i])
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Success Rate (50-episode window)')
        ax3.set_title('Success Rate Evolution')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final Performance Metrics
        ax4 = axes[1, 0]
        final_rewards = [results[param_val]['avg_reward'] for param_val in param_values]
        bars = ax4.bar(range(len(param_values)), final_rewards, alpha=0.7, color=colors)
        ax4.set_xlabel(f'{param_name.upper()} Values')
        ax4.set_ylabel('Final Average Reward')
        ax4.set_title('Final Performance: Average Reward')
        ax4.set_xticks(range(len(param_values)))
        ax4.set_xticklabels([str(v) for v in param_values], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 5: Final Success Rates
        ax5 = axes[1, 1]
        final_success = [results[param_val]['final_success_rate'] for param_val in param_values]
        bars = ax5.bar(range(len(param_values)), final_success, alpha=0.7, color=colors)
        ax5.set_xlabel(f'{param_name.upper()} Values')
        ax5.set_ylabel('Final Success Rate')
        ax5.set_title('Final Performance: Success Rate')
        ax5.set_xticks(range(len(param_values)))
        ax5.set_xticklabels([str(v) for v in param_values], rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Average Steps to Solution
        ax6 = axes[1, 2]
        avg_steps = [results[param_val]['avg_steps'] for param_val in param_values]
        bars = ax6.bar(range(len(param_values)), avg_steps, alpha=0.7, color=colors)
        ax6.set_xlabel(f'{param_name.upper()} Values')
        ax6.set_ylabel('Average Steps to Solution')
        ax6.set_title('Efficiency: Steps per Episode')
        ax6.set_xticks(range(len(param_values)))
        ax6.set_xticklabels([str(v) for v in param_values], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def create_custom_heatmap(self, data, x_labels, y_labels, title, filename):
        """Create a custom heatmap without seaborn"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", color="white")
        
        ax.set_title(title)
        fig.tight_layout()
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Success Rate')
        
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_heatmap_analysis(self, alpha_values, gamma_values, epsilon_values):
        """Create heatmap analysis for combinations of hyperparameters"""
        print("Creating heatmap analysis...")
        
        # Test combinations of alpha and gamma
        print("Testing Alpha vs Gamma combinations...")
        alpha_gamma_results = np.zeros((len(alpha_values), len(gamma_values)))
        
        for i, alpha in enumerate(alpha_values):
            for j, gamma in enumerate(gamma_values):
                print(f"  Testing alpha={alpha}, gamma={gamma}")
                result = self.train_agent(alpha, gamma, epsilon=0.1, episodes=400)
                alpha_gamma_results[i, j] = result['final_success_rate']
        
        # Test combinations of alpha and epsilon
        print("Testing Alpha vs Epsilon combinations...")
        alpha_epsilon_results = np.zeros((len(alpha_values), len(epsilon_values)))
        
        for i, alpha in enumerate(alpha_values):
            for j, epsilon in enumerate(epsilon_values):
                print(f"  Testing alpha={alpha}, epsilon={epsilon}")
                result = self.train_agent(alpha, gamma=0.6, epsilon=epsilon, episodes=400)
                alpha_epsilon_results[i, j] = result['final_success_rate']
        
        # Create heatmaps
        self.create_custom_heatmap(alpha_gamma_results, 
                                 [str(g) for g in gamma_values], 
                                 [str(a) for a in alpha_values],
                                 'Success Rate: Alpha vs Gamma\n(Epsilon=0.1)',
                                 'alpha_gamma_heatmap')
        
        self.create_custom_heatmap(alpha_epsilon_results,
                                 [str(e) for e in epsilon_values],
                                 [str(a) for a in alpha_values],
                                 'Success Rate: Alpha vs Epsilon\n(Gamma=0.6)',
                                 'alpha_epsilon_heatmap')

def evaluate_all_hyperparameters():
    """Main function to evaluate all hyperparameters"""
    evaluator = QLearningHyperparameterEvaluator()
    
    # Define hyperparameter ranges
    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_values = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99]
    epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    episodes = 800
    results_summary = {}
    
    print("="*70)
    print("COMPREHENSIVE Q-LEARNING HYPERPARAMETER EVALUATION")
    print("="*70)
    
    # 1. Evaluate Alpha values
    print("\n1. EVALUATING ALPHA VALUES (Learning Rate)")
    print("-" * 50)
    alpha_results = {}
    for alpha in alpha_values:
        print(f"Training with alpha = {alpha}")
        result = evaluator.train_agent(alpha, gamma=0.6, epsilon=0.1, episodes=episodes)
        alpha_results[alpha] = result
        print(f"  → Success Rate: {result['final_success_rate']:.3f}, Avg Reward: {result['avg_reward']:.2f}")
    
    evaluator.plot_hyperparameter_comparison('alpha', alpha_values, alpha_results, 'alpha_analysis')
    results_summary['alpha'] = alpha_results
    
    # 2. Evaluate Gamma values  
    print("\n2. EVALUATING GAMMA VALUES (Discount Factor)")
    print("-" * 50)
    gamma_results = {}
    for gamma in gamma_values:
        print(f"Training with gamma = {gamma}")
        result = evaluator.train_agent(alpha=0.1, gamma=gamma, epsilon=0.1, episodes=episodes)
        gamma_results[gamma] = result
        print(f"  → Success Rate: {result['final_success_rate']:.3f}, Avg Reward: {result['avg_reward']:.2f}")
    
    evaluator.plot_hyperparameter_comparison('gamma', gamma_values, gamma_results, 'gamma_analysis')
    results_summary['gamma'] = gamma_results
    
    # 3. Evaluate Epsilon values
    print("\n3. EVALUATING EPSILON VALUES (Exploration Rate)")
    print("-" * 50)
    epsilon_results = {}
    for epsilon in epsilon_values:
        print(f"Training with epsilon = {epsilon}")
        result = evaluator.train_agent(alpha=0.1, gamma=0.6, epsilon=epsilon, episodes=episodes)
        epsilon_results[epsilon] = result
        print(f"  → Success Rate: {result['final_success_rate']:.3f}, Avg Reward: {result['avg_reward']:.2f}")
    
    evaluator.plot_hyperparameter_comparison('epsilon', epsilon_values, epsilon_results, 'epsilon_analysis')
    results_summary['epsilon'] = epsilon_results
    
    # 4. Create heatmap analysis (reduced values for computational efficiency)
    print("\n4. CREATING HEATMAP ANALYSIS")
    print("-" * 50)
    alpha_heatmap = [0.05, 0.1, 0.3, 0.5, 0.7]
    gamma_heatmap = [0.3, 0.6, 0.8, 0.9, 0.99]
    epsilon_heatmap = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    evaluator.create_heatmap_analysis(alpha_heatmap, gamma_heatmap, epsilon_heatmap)
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    # Find best parameters for each hyperparameter
    best_alpha = max(alpha_results.keys(), key=lambda k: alpha_results[k]['final_success_rate'])
    best_gamma = max(gamma_results.keys(), key=lambda k: gamma_results[k]['final_success_rate'])
    best_epsilon = max(epsilon_results.keys(), key=lambda k: epsilon_results[k]['final_success_rate'])
    
    print(f"\n🏆 BEST HYPERPARAMETERS:")
    print(f"   Alpha (Learning Rate):     {best_alpha:6.3f} → Success: {alpha_results[best_alpha]['final_success_rate']:.3f}")
    print(f"   Gamma (Discount Factor):   {best_gamma:6.3f} → Success: {gamma_results[best_gamma]['final_success_rate']:.3f}")
    print(f"   Epsilon (Exploration):     {best_epsilon:6.3f} → Success: {epsilon_results[best_epsilon]['final_success_rate']:.3f}")
    
    print(f"\n📊 HYPERPARAMETER INSIGHTS:")
    
    # Alpha insights
    alpha_performance = [(k, v['final_success_rate']) for k, v in alpha_results.items()]
    alpha_performance.sort(key=lambda x: x[1], reverse=True)
    print(f"   Alpha Rankings: {', '.join([f'{a}({s:.3f})' for a, s in alpha_performance[:3]])}")
    
    # Gamma insights
    gamma_performance = [(k, v['final_success_rate']) for k, v in gamma_results.items()]
    gamma_performance.sort(key=lambda x: x[1], reverse=True)
    print(f"   Gamma Rankings: {', '.join([f'{g}({s:.3f})' for g, s in gamma_performance[:3]])}")
    
    # Epsilon insights
    epsilon_performance = [(k, v['final_success_rate']) for k, v in epsilon_results.items()]
    epsilon_performance.sort(key=lambda x: x[1], reverse=True)
    print(f"   Epsilon Rankings: {', '.join([f'{e}({s:.3f})' for e, s in epsilon_performance[:3]])}")
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION:")
    print(f"   alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon}")
    print(f"   Expected Performance: ~{max(alpha_results[best_alpha]['final_success_rate'], gamma_results[best_gamma]['final_success_rate'], epsilon_results[best_epsilon]['final_success_rate']):.1%} success rate")
    
    return results_summary

if __name__ == "__main__":
    results = evaluate_all_hyperparameters()