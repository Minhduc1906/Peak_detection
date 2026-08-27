import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set up styling
plt.style.use('default')
sns.set_palette("husl")

def create_output_directory():
    """Create the output directory if it doesn't exist"""
    output_dir = Path("results/time_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_data():
    """Load all CSV files and return organized data"""
    
    # Load the latest files
    try:
        # Parameter test data
        param_file_timing = pd.read_csv("results/time/latest_param_test_file_timing.csv")
        param_overall_timing = pd.read_csv("results/time/latest_param_test_overall_timing.csv")
        
        # Training data  
        training_times = pd.read_csv("results/time/latest_training_times.csv")
        
        print("Data loaded successfully!")
        print(f"Parameter file timing: {len(param_file_timing)} rows")
        print(f"Parameter overall timing: {len(param_overall_timing)} rows") 
        print(f"Training times: {len(training_times)} rows")
        
        return param_file_timing, param_overall_timing, training_times
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def categorize_models(training_times):
    """Categorize models into finetune, combined, and identify heuristic from param data"""
    
    # Print unique model names to understand the data
    if 'model_name' in training_times.columns:
        print("\nUnique model names in training data:")
        print(training_times['model_name'].unique())
    
    # Categorize based on model names (adjust these based on your actual model names)
    finetune_models = training_times[
        training_times['model_name'].str.contains('finetune|fine_tune|ft_', case=False, na=False)
    ].copy()
    
    combined_models = training_times[
        training_times['model_name'].str.contains('combined|combine|ensemble', case=False, na=False)
    ].copy()
    
    # If the above doesn't work, we might need to use other criteria
    if len(finetune_models) == 0 and len(combined_models) == 0:
        # Alternative: use is_finetuned column if available
        if 'is_finetuned' in training_times.columns:
            finetune_models = training_times[training_times['is_finetuned'] == 'True'].copy()
            combined_models = training_times[training_times['is_finetuned'] == 'False'].copy()
        else:
            # Fallback: split the data arbitrarily for demonstration
            print("Warning: Could not automatically categorize models. Using fallback method.")
            mid_point = len(training_times) // 2
            finetune_models = training_times[:mid_point].copy()
            combined_models = training_times[mid_point:].copy()
    
    return finetune_models, combined_models

def calculate_metrics(param_overall_timing, param_file_timing, finetune_models, combined_models):
    """Calculate the required metrics for comparison"""
    
    metrics = {}
    
    # Heuristic param test metrics
    # Total time from overall timing (process-level)
    heuristic_data = param_overall_timing[
        param_overall_timing['process_name'].str.contains('param|heuristic', case=False, na=False)
    ]
    
    if len(heuristic_data) > 0:
        heuristic_total_time = heuristic_data['duration_seconds'].sum()
    else:
        # Use all param test data if no specific heuristic label
        heuristic_total_time = param_overall_timing['duration_seconds'].sum()
    
    # Average time per test case = total time / number of individual test files
    num_param_test_cases = len(param_file_timing)
    heuristic_avg_time = heuristic_total_time / num_param_test_cases if num_param_test_cases > 0 else 0
    
    # Time per iteration (12600 parameter combinations)
    heuristic_time_per_iteration = heuristic_total_time / 12600
    
    # Debug print for verification
    print(f"\nHeuristic calculations:")
    print(f"• Total time: {heuristic_total_time:.1f} seconds")
    print(f"• Number of test cases: {num_param_test_cases}")
    print(f"• Average time per test case: {heuristic_avg_time:.1f} seconds")
    print(f"• Verification: {heuristic_avg_time:.1f} × {num_param_test_cases} = {heuristic_avg_time * num_param_test_cases:.1f} seconds")
    
    # Finetune model metrics
    if len(finetune_models) > 0:
        finetune_total_time = finetune_models['duration_seconds'].sum()
        finetune_avg_time = finetune_models['duration_seconds'].mean()
        # Time per epoch (20 epochs)
        finetune_time_per_iteration = finetune_total_time / (len(finetune_models) * 20)
        
        print(f"\nFinetune calculations:")
        print(f"• Total time: {finetune_total_time:.1f} seconds")
        print(f"• Number of test cases: {len(finetune_models)}")
        print(f"• Average time per test case: {finetune_avg_time:.1f} seconds")
        print(f"• Verification: {finetune_avg_time:.1f} × {len(finetune_models)} = {finetune_avg_time * len(finetune_models):.1f} seconds")
    else:
        finetune_total_time = finetune_avg_time = finetune_time_per_iteration = 0
        print("\nFinetune calculations: No finetune models found")
    
    # Combined model metrics
    if len(combined_models) > 0:
        combined_total_time = combined_models['duration_seconds'].sum()
        combined_avg_time = combined_models['duration_seconds'].mean()
        # Time per epoch (20 epochs)
        combined_time_per_iteration = combined_total_time / (len(combined_models) * 20)
        
        print(f"\nCombined calculations:")
        print(f"• Total time: {combined_total_time:.1f} seconds")
        print(f"• Number of test cases: {len(combined_models)}")
        print(f"• Average time per test case: {combined_avg_time:.1f} seconds")
        print(f"• Verification: {combined_avg_time:.1f} × {len(combined_models)} = {combined_avg_time * len(combined_models):.1f} seconds")
    else:
        combined_total_time = combined_avg_time = combined_time_per_iteration = 0
        print("\nCombined calculations: No combined models found")
    
    metrics = {
        'model_types': ['Finetune', 'Heuristic', 'Combined'],
        'total_times': [finetune_total_time, heuristic_total_time, combined_total_time],
        'avg_times': [finetune_avg_time, heuristic_avg_time, combined_avg_time],
        'time_per_iteration': [finetune_time_per_iteration, heuristic_time_per_iteration, combined_time_per_iteration],
        'test_counts': [len(finetune_models), num_param_test_cases, len(combined_models)],
        'colors': ['green', 'red', 'purple']
    }
    
    return metrics

def create_comparison_graphs(metrics, output_dir):
    """Create three comparison graphs"""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Graph 1: Total Time Cost
    axes[0].bar(metrics['model_types'], metrics['total_times'], color=metrics['colors'], alpha=0.7)
    axes[0].set_title('Total Time Cost for All Training', fontweight='bold')
    axes[0].set_ylabel('Total Time (seconds)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(metrics['total_times']):
        axes[0].text(i, v + max(metrics['total_times']) * 0.01, f'{v:.1f}s', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Graph 2: Time per Training Session (only for models with multiple sessions)
    # Filter out combined model for this comparison since it has only 1 session
    filtered_types = []
    filtered_avg_times = []
    filtered_colors = []
    filtered_test_counts = []
    
    for i, (model_type, avg_time, color, test_count) in enumerate(zip(
        metrics['model_types'], metrics['avg_times'], metrics['colors'], metrics['test_counts'])):
        if test_count > 1:  # Only include models with multiple training sessions
            filtered_types.append(f"{model_type}\n({test_count} sessions)")
            filtered_avg_times.append(avg_time)
            filtered_colors.append(color)
    
    if filtered_avg_times:
        axes[1].bar(filtered_types, filtered_avg_times, color=filtered_colors, alpha=0.7)
        axes[1].set_title('Average Time per Training Session\n(Multiple Session Models Only)', fontweight='bold')
        axes[1].set_ylabel('Average Time (seconds)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(filtered_avg_times):
            axes[1].text(i, v + max(filtered_avg_times) * 0.01, f'{v:.1f}s', 
                        ha='center', va='bottom', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No models with\nmultiple sessions', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Average Time per Training Session', fontweight='bold')
    
    # Graph 3: Time Cost per Training Iteration
    axes[2].bar(metrics['model_types'], metrics['time_per_iteration'], color=metrics['colors'], alpha=0.7)
    axes[2].set_title('Time Cost per Training Iteration', fontweight='bold')
    axes[2].set_ylabel('Time per Iteration (seconds)')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(metrics['time_per_iteration']):
        axes[2].text(i, v + max(metrics['time_per_iteration']) * 0.01, f'{v:.3f}s', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / "model_time_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison graph saved to: {output_path}")
    
    # Also create individual graphs for better visibility
    create_individual_graphs(metrics, output_dir)
    
    plt.show()

def create_individual_graphs(metrics, output_dir):
    """Create individual graphs for each metric"""
    
    # Individual graph 1: Total Time Cost
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(metrics['model_types'], metrics['total_times'], color=metrics['colors'], alpha=0.8)
    plt.title('Total Time Cost for All Training', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add value labels
    for i, v in enumerate(metrics['total_times']):
        plt.text(i, v + max(metrics['total_times']) * 0.01, f'{v:.1f}s', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "total_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual graph 2: Average Time Cost (only for multi-session models)
    filtered_types = []
    filtered_avg_times = []
    filtered_colors = []
    
    for i, (model_type, avg_time, color, test_count) in enumerate(zip(
        metrics['model_types'], metrics['avg_times'], metrics['colors'], metrics['test_counts'])):
        if test_count > 1:
            filtered_types.append(f"{model_type}\n({test_count} sessions)")
            filtered_avg_times.append(avg_time)
            filtered_colors.append(color)
    
    if filtered_avg_times:
        plt.figure(figsize=(10, 6))
        bars2 = plt.bar(filtered_types, filtered_avg_times, color=filtered_colors, alpha=0.8)
        plt.title('Average Time per Training Session\n(Multiple Session Models Only)', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Average Time (seconds)', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add value labels
        for i, v in enumerate(filtered_avg_times):
            plt.text(i, v + max(filtered_avg_times) * 0.01, f'{v:.1f}s', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "average_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Create a placeholder graph if no multi-session models
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No models with multiple training sessions\nfor meaningful average comparison', 
                ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title('Average Time per Training Session', fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "average_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Individual graph 3: Time per Iteration
    plt.figure(figsize=(10, 6))
    bars3 = plt.bar(metrics['model_types'], metrics['time_per_iteration'], color=metrics['colors'], alpha=0.8)
    plt.title('Time Cost per Training Iteration', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Time per Iteration (seconds)', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add value labels
    for i, v in enumerate(metrics['time_per_iteration']):
        plt.text(i, v + max(metrics['time_per_iteration']) * 0.01, f'{v:.3f}s', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "time_per_iteration_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_table(metrics):
    """Print a summary table of the metrics"""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Model Type': metrics['model_types'],
        'Sessions': metrics['test_counts'],
        'Total Time (s)': [f"{t:.1f}" for t in metrics['total_times']],
        'Avg Time per Session (s)': [f"{t:.1f}" for t in metrics['avg_times']],
        'Time per Iteration (s)': [f"{t:.3f}" for t in metrics['time_per_iteration']]
    })
    
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Additional insights
    print("\nKey Insights:")
    best_total = np.argmin(metrics['total_times'])
    best_iteration = np.argmin(metrics['time_per_iteration'])
    
    # Only compare average times for models with multiple sessions
    multi_session_indices = [i for i, count in enumerate(metrics['test_counts']) if count > 1]
    if multi_session_indices:
        multi_session_avg_times = [metrics['avg_times'][i] for i in multi_session_indices]
        best_avg_idx = multi_session_indices[np.argmin(multi_session_avg_times)]
        print(f"• Fastest total time: {metrics['model_types'][best_total]} ({metrics['total_times'][best_total]:.1f}s)")
        print(f"• Fastest average time (multi-session models): {metrics['model_types'][best_avg_idx]} ({metrics['avg_times'][best_avg_idx]:.1f}s)")
        print(f"• Fastest per iteration: {metrics['model_types'][best_iteration]} ({metrics['time_per_iteration'][best_iteration]:.3f}s)")
    else:
        print(f"• Fastest total time: {metrics['model_types'][best_total]} ({metrics['total_times'][best_total]:.1f}s)")
        print(f"• Fastest per iteration: {metrics['model_types'][best_iteration]} ({metrics['time_per_iteration'][best_iteration]:.3f}s)")
        print("• No models with multiple sessions for average comparison")
    
    print("\nNote:")
    print("• 'Sessions' = Number of separate training/testing runs")
    print("• Combined model trains once on entire dataset (1 session)")
    print("• Average time per session only meaningful for models with multiple sessions")

def main():
    """Main function to run the analysis"""
    
    print("Starting Model Time Cost Comparison Analysis...")
    print("-" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    
    # Load data
    param_file_timing, param_overall_timing, training_times = load_data()
    
    if param_overall_timing is None or training_times is None:
        print("Error: Could not load required data files.")
        return
    
    # Categorize models
    finetune_models, combined_models = categorize_models(training_times)
    
    print(f"\nModel categorization:")
    print(f"• Finetune models: {len(finetune_models)} entries")
    print(f"• Combined models: {len(combined_models)} entries")
    print(f"• Heuristic param tests: {len(param_overall_timing)} entries")
    
    # Calculate metrics
    metrics = calculate_metrics(param_overall_timing, param_file_timing, finetune_models, combined_models)
    
    # Create graphs
    create_comparison_graphs(metrics, output_dir)
    
    # Print summary
    print_summary_table(metrics)
    
    print(f"\nAnalysis complete! Graphs saved to: {output_dir}")
    print("Files generated:")
    print("• model_time_comparison.png (combined view)")
    print("• total_time_comparison.png")
    print("• average_time_comparison.png (multi-session models only)") 
    print("• time_per_iteration_comparison.png")
    print("\nNote: Average time comparison excludes single-session models (like combined)")
    print("      since 'average per session' = 'total time' for single sessions.")

if __name__ == "__main__":
    main()