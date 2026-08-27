import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.signal import medfilt
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

def peak_speed_detect(throughput, time, long_window_ms=2000, short_window_ms=200, peak_tolerance=0.1, count_threshold=5):
    if len(throughput) != len(time):
        raise ValueError("throughput and time arrays must have same length")
    if len(throughput) < 2 or np.all(throughput == 0):
        print("Warning: Throughput is zero or too short. No clipping detected.")
        return np.zeros(len(throughput), dtype=int), np.zeros(len(throughput)), throughput.copy(), np.zeros(len(throughput)), np.zeros(len(throughput))
    
    mean_dt = np.mean(np.diff(time))
    long_window_samples = max(1, int(long_window_ms / 1000 / mean_dt))
    short_window_samples = max(1, int(short_window_ms / 1000 / mean_dt))
    
    n_samples = len(throughput)
    filter_length = min(15, n_samples // 2 * 2 + 1)  # Ensure odd
    filtered_throughput = medfilt(throughput, filter_length)
    
    clipping_binary = np.zeros(n_samples, dtype=int)
    long_maxima = np.zeros(n_samples)
    short_counts = np.zeros(n_samples)
    short_peaks = np.zeros(n_samples)
    
    for long_start in range(0, n_samples, long_window_samples):
        long_end = min(long_start + long_window_samples, n_samples)
        long_window_data = throughput[long_start:long_end]
        if len(long_window_data) == 0:
            continue
        long_max = np.max(long_window_data)
        long_maxima[long_start:long_end] = long_max
        
        for short_start in range(long_start, long_end, short_window_samples):
            short_end = min(short_start + short_window_samples, long_end)
            short_bin_data = throughput[short_start:short_end]
            if len(short_bin_data) == 0:
                continue
            short_peak = np.max(short_bin_data)
            short_peaks[short_start:short_end] = short_peak
            tolerance_threshold = (1 - peak_tolerance) * long_max if long_max > 0 else 0
            near_peak_count = np.sum(short_bin_data >= tolerance_threshold)
            short_counts[short_start:short_end] = near_peak_count
            if long_max > 0 and near_peak_count >= count_threshold:
                clipping_binary[short_start:short_end] = 1
    
    peak_ratio = np.zeros(n_samples)
    nonzero_mask = long_maxima > 0
    peak_ratio[nonzero_mask] = short_peaks[nonzero_mask] / long_maxima[nonzero_mask]
    
    return (clipping_binary, long_maxima, filtered_throughput, 
            np.nan_to_num(peak_ratio, nan=0.0, posinf=0.0, neginf=0.0), 
            np.nan_to_num(short_counts, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(short_peaks, nan=0.0, posinf=0.0, neginf=0.0))

def calculate_throughput_matlab_method(df, byte_col='tx_bytes', time_col='relative_time'):
    """
    Calculate throughput using MATLAB-compatible method (consistent with feature extraction)
    MATLAB: 8 * ds(2:end, 5) / ds_interval / 1e6
    
    This method treats byte values as rates per interval, not cumulative counters.
    """
    if byte_col not in df.columns:
        print(f"Warning: {byte_col} not in columns")
        return np.zeros(len(df)), df[time_col].values
    
    time_values = df[time_col].values
    if len(time_values) < 3:
        print(f"Warning: Insufficient time points ({len(time_values)})")
        return np.zeros(len(df)), time_values
    
    # MATLAB: t_ds = ds(2:end, 1) - ds(2, 1)  (relative time from second row)
    relative_time = time_values[1:] - time_values[1]  # Start from second row
    if len(relative_time) < 2:
        print(f"Warning: Cannot calculate interval from {len(relative_time)} points")
        return np.zeros(len(df)), time_values
    
    # MATLAB: ds_interval = mean (t_ds(2:end) - t_ds(1:end-1))
    interval = np.mean(np.diff(relative_time))
    if interval <= 0:
        print(f"Warning: Invalid interval {interval}, using default 0.02")
        interval = 0.02
    
    # MATLAB: 8 * ds(2:end, 5) / ds_interval / 1e6
    byte_values = df[byte_col].values[1:]  # Skip first row like MATLAB
    
    # Check for valid data
    if df[byte_col].nunique() <= 1 or df[byte_col].isna().all():
        print(f"Warning: {byte_col} is constant or all NaN")
        return np.zeros(len(df)), time_values
    
    # Handle NaN values
    if np.any(np.isnan(byte_values)):
        print(f"Warning: {byte_col} contains NaNs. Filling with 0")
        byte_values = np.nan_to_num(byte_values, nan=0.0)
    
    # Calculate Mbps: convert bytes to bits, normalize by interval, convert to Mbps
    bit_rate = 8 * byte_values / interval / 1e6
    
    # Handle negative values (counter resets) by setting to 0
    bit_rate = np.maximum(bit_rate, 0)
    
    # Pad with zero at the beginning to match original dataframe length
    bit_rate_full = np.concatenate([np.zeros(1), bit_rate])
    
    # Ensure no inf/nan values
    bit_rate_full = np.nan_to_num(bit_rate_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    return bit_rate_full, time_values

def calculate_throughput(df, byte_col='tx_bytes', fallback_byte_col='rx_bytes', time_col='relative_time'):
    """
    Updated to use MATLAB-compatible throughput calculation for consistency
    """
    try:
        # Find time columns if time_col doesn't exist
        if time_col not in df.columns:
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
            else:
                df['index_time'] = df.index.astype(float)
                time_col = 'index_time'
        
        # Try primary byte column first
        if byte_col in df.columns and df[byte_col].nunique() > 1 and not df[byte_col].isna().all():
            return calculate_throughput_matlab_method(df, byte_col, time_col)
        
        # Fallback to secondary byte column
        elif fallback_byte_col in df.columns and df[fallback_byte_col].nunique() > 1 and not df[fallback_byte_col].isna().all():
            print(f"Warning: {byte_col} not usable. Using fallback {fallback_byte_col}")
            return calculate_throughput_matlab_method(df, fallback_byte_col, time_col)
        
        else:
            print(f"Error: Neither {byte_col} nor {fallback_byte_col} contain valid data")
            time_values = df[time_col].values if time_col in df.columns else np.arange(len(df))
            return np.zeros(len(df)), time_values
            
    except Exception as e:
        print(f"Error in calculate_throughput: {str(e)}")
        time_values = df[time_col].values if time_col in df.columns else np.arange(len(df))
        return np.zeros(len(df)), time_values

# Setup - Increase font sizes globally
os.makedirs('enhanced_parameter_analysis', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.4)  # Increased from 1.0
plt.rcParams.update({'font.size': 14})  # Increased base font size

# Load data
try:
    data = pd.read_csv('combined_param_tests.csv')
except FileNotFoundError:
    print("Error: combined_param_tests.csv not found.")
    exit()

# Filter downstream data
data = data[data['direction'] == 'downstream']
print(f"Downstream records: {len(data)}")

# Get successful tests with all metrics
success_data = data[(data['status'] == 'success') & 
                   (data['accuracy'].notna()) & 
                   (data['precision'].notna()) & 
                   (data['recall'].notna()) & 
                   (data['f1_score'].notna())].copy()

print(f"Successful tests with all metrics: {len(success_data)}")
if len(success_data) == 0:
    print("No successful downstream tests found with all metrics.")
    exit()

# Create test case identifiers
success_data['file_name_clean'] = success_data['file_name'].str.replace('.csv', '', regex=False)
success_data['test_case'] = (success_data['file_name_clean'] + '_' + success_data['direction'] + '_' + 
                             success_data['subfolder'] + '_' + success_data['network_limit'].astype(str))

# Group data by parameter combinations with all metrics
metrics = ['accuracy', 'precision', 'recall', 'f1_score']  # Fixed: consistent naming
grouped_data = success_data.groupby(['short_window', 'long_window', 'peak_tolerance', 'count_threshold']).agg({
    'accuracy': ['mean', 'std', 'min', 'max', 'count'],
    'precision': ['mean', 'std', 'min', 'max'],
    'recall': ['mean', 'std', 'min', 'max'],
    'f1_score': ['mean', 'std', 'min', 'max']
}).reset_index()

# Flatten column names
grouped_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_data.columns.values]
grouped_data.rename(columns={'accuracy_count': 'test_count'}, inplace=True)

print(f"Unique parameter combinations: {len(grouped_data)}")
print("✅ Now using MATLAB-compatible throughput calculation for consistency")

# ===========================================
# SECTION 1: Multi-Metric Overview
# ===========================================

def plot_metric_distributions():
    """Plot distributions of all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increased figure size
    axes = axes.flatten()
    
    metrics_plot = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_score_mean']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (metric, name, color) in enumerate(zip(metrics_plot, metric_names, colors)):
        axes[i].hist(grouped_data[metric], bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[i].set_title(f'{name} Distribution', fontsize=16, fontweight='bold')
        axes[i].set_xlabel(f'{name} Score', fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
        axes[i].axvline(grouped_data[metric].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {grouped_data[metric].mean():.3f}')
        axes[i].legend(fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/1_metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_metric_distributions()

# ===========================================
# SECTION 2: Parameter Pair Analysis (All Metrics)
# ===========================================

def create_metric_heatmaps():
    """Create heatmaps for all metrics across parameter pairs"""
    
    param_pairs = [
        ('short_window', 'long_window', 'Short Window (ms)', 'Long Window (ms)'),
        ('short_window', 'peak_tolerance', 'Short Window (ms)', 'Peak Tolerance'),
        ('short_window', 'count_threshold', 'Short Window (ms)', 'Count Threshold'),
        ('long_window', 'peak_tolerance', 'Long Window (ms)', 'Peak Tolerance'),
        ('long_window', 'count_threshold', 'Long Window (ms)', 'Count Threshold'),
        ('peak_tolerance', 'count_threshold', 'Peak Tolerance', 'Count Threshold')
    ]
    
    for pair_idx, (param1, param2, label1, label2) in enumerate(param_pairs):
        fig, axes = plt.subplots(2, 2, figsize=(22, 18))  # Increased figure size
        fig.suptitle(f'All Metrics: {label1} vs {label2}', fontsize=18, fontweight='bold')
        
        for metric_idx, (metric, name) in enumerate(zip(metrics, ['Accuracy', 'Precision', 'Recall', 'F1-Score'])):
            ax = axes[metric_idx // 2, metric_idx % 2]
            
            pivot = grouped_data.pivot_table(index=param1, columns=param2, values=f'{metric}_mean', aggfunc='mean')
            
            im = sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f', ax=ax,
                           cbar_kws={'label': f'{name} Score'}, annot_kws={'size': 12})
            ax.set_title(f'{name} by {label1} vs {label2}', fontsize=16, fontweight='bold')
            ax.set_xlabel(label2, fontsize=14)
            ax.set_ylabel(label1, fontsize=14)
            ax.tick_params(labelsize=12)
            
            # Highlight best performance
            if not pivot.empty:
                max_val = pivot.max().max()
                max_pos = np.where(pivot.values == max_val)
                if len(max_pos[0]) > 0:
                    ax.add_patch(plt.Rectangle((max_pos[1][0], max_pos[0][0]), 1, 1, 
                                             fill=False, edgecolor='red', lw=4))
        
        plt.tight_layout()
        plt.savefig(f'enhanced_parameter_analysis/2_{pair_idx+1}_heatmap_{param1}_{param2}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

create_metric_heatmaps()

# ===========================================
# SECTION 3: 4D Parameter Space Analysis
# ===========================================

def analyze_parameter_importance():
    """Analyze importance of each parameter for each metric"""
    
    params = ['short_window', 'long_window', 'peak_tolerance', 'count_threshold']
    param_labels = ['Short Window', 'Long Window', 'Peak Tolerance', 'Count Threshold']
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))  # Increased figure size
    fig.suptitle('Parameter Importance Analysis (Variance Explained)', fontsize=18, fontweight='bold')
    
    for metric_idx, (metric, name) in enumerate(zip(metrics, ['Accuracy', 'Precision', 'Recall', 'F1-Score'])):
        ax = axes[metric_idx // 2, metric_idx % 2]
        
        # Calculate variance explained by each parameter
        importance_scores = []
        for param in params:
            param_means = grouped_data.groupby(param)[f'{metric}_mean'].mean()
            overall_mean = grouped_data[f'{metric}_mean'].mean()
            variance_explained = np.sum((param_means - overall_mean) ** 2) / len(param_means)
            importance_scores.append(variance_explained)
        
        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        importance_scores = importance_scores / np.sum(importance_scores) * 100
        
        # Create bar plot
        bars = ax.bar(param_labels, importance_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax.set_title(f'{name} - Parameter Importance', fontsize=16, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=14)
        ax.set_ylim(0, max(importance_scores) * 1.3)  # More space for labels
        
        # Add value labels on bars - Fixed positioning
        for bar, score in zip(bars, importance_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(importance_scores)*0.02,
                   f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/3_parameter_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

analyze_parameter_importance()

# ===========================================
# SECTION 4: Best Parameter Combinations by Metric - FIXED
# ===========================================

def find_best_combinations():
    """Find best parameter combinations for each metric and overall"""
    
    results = {}
    
    # Best by individual metrics
    for metric, name in zip(metrics, ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        best_row = grouped_data.loc[grouped_data[f'{metric}_mean'].idxmax()]
        results[f'best_{metric}'] = {
            'name': name,
            'short_window': best_row['short_window'],
            'long_window': best_row['long_window'],
            'peak_tolerance': best_row['peak_tolerance'],
            'count_threshold': best_row['count_threshold'],
            'score': best_row[f'{metric}_mean'],
            'test_count': best_row['test_count']
        }
    
    # Best balanced (highest average across all metrics)
    grouped_data['balanced_score'] = (grouped_data['accuracy_mean'] + 
                                     grouped_data['precision_mean'] + 
                                     grouped_data['recall_mean'] + 
                                     grouped_data['f1_score_mean']) / 4
    
    best_balanced = grouped_data.loc[grouped_data['balanced_score'].idxmax()]
    results['best_balanced'] = {
        'name': 'Balanced',
        'short_window': best_balanced['short_window'],
        'long_window': best_balanced['long_window'],
        'peak_tolerance': best_balanced['peak_tolerance'],
        'count_threshold': best_balanced['count_threshold'],
        'score': best_balanced['balanced_score'],
        'test_count': best_balanced['test_count'],
        'accuracy': best_balanced['accuracy_mean'],
        'precision': best_balanced['precision_mean'],
        'recall': best_balanced['recall_mean'],
        'f1_score': best_balanced['f1_score_mean']
    }
    
    # Create visualization - FIXED with parameter combinations shown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # Increased figure size
    
    # Plot 1: Best combinations comparison with parameter combinations
    combinations = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Balanced']
    scores = [results[f'best_{metric}']['score'] for metric in metrics] + [results['best_balanced']['score']]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    bars = ax1.bar(combinations, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_title('Best Parameter Combinations by Optimization Target', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=14)
    ax1.set_ylim(0, 1.1)  # Increased ylim for parameter text
    
    # Add score labels and parameter combinations - FIXED
    for i, (bar, score, combo_name) in enumerate(zip(bars, scores, combinations)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add parameter combination below the bar
        if combo_name == 'Balanced':
            combo_info = results['best_balanced']
        else:
            combo_info = results[f'best_{metrics[i]}']
        
        param_text = f"SW:{combo_info['short_window']}\nLW:{combo_info['long_window']}\nPT:{combo_info['peak_tolerance']}\nCT:{combo_info['count_threshold']}"
        ax1.text(bar.get_x() + bar.get_width()/2., -0.15,  # Position below x-axis
                param_text, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Balanced combination detailed scores
    balanced_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    balanced_scores = [results['best_balanced'][metric.lower().replace('-', '_')] for metric in balanced_metrics]
    
    bars2 = ax2.bar(balanced_metrics, balanced_scores, color='purple', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_title('Best Balanced Combination - All Metrics', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=14)
    ax2.set_ylim(0, 1.1)
    
    for bar, score in zip(bars2, balanced_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add parameter combination for balanced
    combo_info = results['best_balanced']
    param_text = f"SW:{combo_info['short_window']}, LW:{combo_info['long_window']}, PT:{combo_info['peak_tolerance']}, CT:{combo_info['count_threshold']}"
    ax2.text(0.5, 0.9, param_text, transform=ax2.transAxes, ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/4_best_combinations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

best_combinations = find_best_combinations()

# ===========================================
# SECTION 5: Parameter Sensitivity Analysis
# ===========================================

def parameter_sensitivity_analysis():
    """Analyze how each parameter affects each metric"""
    
    params = ['short_window', 'long_window', 'peak_tolerance', 'count_threshold']
    param_labels = ['Short Window (ms)', 'Long Window (ms)', 'Peak Tolerance', 'Count Threshold']
    
    fig, axes = plt.subplots(len(params), len(metrics), figsize=(24, 20))  # Increased figure size
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=18, fontweight='bold')
    
    for param_idx, (param, param_label) in enumerate(zip(params, param_labels)):
        for metric_idx, (metric, metric_name) in enumerate(zip(metrics, ['Accuracy', 'Precision', 'Recall', 'F1-Score'])):
            ax = axes[param_idx, metric_idx]
            
            # Calculate mean and std for each parameter value
            param_analysis = grouped_data.groupby(param).agg({
                f'{metric}_mean': ['mean', 'std', 'count']
            }).reset_index()
            param_analysis.columns = [param, 'mean', 'std', 'count']
            param_analysis = param_analysis.sort_values(param)
            
            # Plot mean with error bars
            ax.errorbar(param_analysis[param], param_analysis['mean'], 
                       yerr=param_analysis['std'], marker='o', capsize=5, capthick=2, markersize=8)
            
            # Add trend line
            z = np.polyfit(param_analysis[param], param_analysis['mean'], 1)
            p = np.poly1d(z)
            ax.plot(param_analysis[param], p(param_analysis[param]), "--", alpha=0.7, color='red', linewidth=2)
            
            ax.set_title(f'{metric_name} vs {param_label}', fontsize=14, fontweight='bold')
            ax.set_xlabel(param_label, fontsize=12)
            ax.set_ylabel(f'{metric_name} Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)
            
            # Highlight best value
            best_idx = param_analysis['mean'].idxmax()
            best_val = param_analysis.iloc[best_idx]
            ax.scatter(best_val[param], best_val['mean'], color='red', s=150, zorder=5)
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/5_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

parameter_sensitivity_analysis()

# ===========================================
# SECTION 6: Robust Parameter Selection
# ===========================================

def robust_parameter_analysis():
    """Find parameters that are consistently good across different conditions"""
    
    # Calculate robustness score (mean - penalty for high variance)
    for metric in metrics:
        grouped_data[f'{metric}_robustness'] = (grouped_data[f'{metric}_mean'] - 
                                              0.5 * grouped_data[f'{metric}_std'])
    
    # Overall robustness score
    grouped_data['overall_robustness'] = (grouped_data['accuracy_robustness'] + 
                                        grouped_data['precision_robustness'] + 
                                        grouped_data['recall_robustness'] + 
                                        grouped_data['f1_score_robustness']) / 4
    
    # Get top robust combinations
    top_robust = grouped_data.nlargest(20, 'overall_robustness')
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # Increased figure size
    
    # Plot 1: Robustness vs Performance scatter
    scatter = ax1.scatter(grouped_data['f1_score_mean'], grouped_data['overall_robustness'], 
                         c=grouped_data['test_count'], s=60, alpha=0.6, cmap='viridis')
    ax1.set_xlabel('F1-Score Mean', fontsize=14)
    ax1.set_ylabel('Overall Robustness Score', fontsize=14)
    ax1.set_title('Performance vs Robustness Trade-off', fontsize=16, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Test Count')
    ax1.tick_params(labelsize=12)
    
    # Highlight top robust combinations
    for idx, row in top_robust.head(5).iterrows():
        ax1.scatter(row['f1_score_mean'], row['overall_robustness'], 
                   color='red', s=120, alpha=0.8)
        ax1.annotate(f"SW:{row['short_window']}, LW:{row['long_window']}\nPT:{row['peak_tolerance']}, CT:{row['count_threshold']}", 
                    (row['f1_score_mean'], row['overall_robustness']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Top robust combinations
    top_10_robust = top_robust.head(10)
    x_pos = np.arange(len(top_10_robust))
    
    bars = ax2.bar(x_pos, top_10_robust['overall_robustness'], color='skyblue', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_title('Top 10 Most Robust Parameter Combinations', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Parameter Combination Rank', fontsize=14)
    ax2.set_ylabel('Overall Robustness Score', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i+1}" for i in range(len(top_10_robust))], rotation=0)
    ax2.tick_params(labelsize=12)
    
    # Add parameter labels - Fixed positioning
    for i, (idx, row) in enumerate(top_10_robust.iterrows()):
        height = bars[i].get_height()
        ax2.text(i, height + 0.01, 
                f"SW:{row['short_window']}\nLW:{row['long_window']}\nPT:{row['peak_tolerance']}\nCT:{row['count_threshold']}", 
                ha='center', va='bottom', fontsize=9, rotation=0,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/6_robust_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_robust

top_robust_combinations = robust_parameter_analysis()

# ===========================================
# SECTION 7: Heuristic Recommendations
# ===========================================

def generate_heuristic_recommendations():
    """Generate heuristic recommendations for different use cases"""
    
    recommendations = {}
    
    # High Precision (minimize false positives)
    high_precision = grouped_data.nlargest(10, 'precision_mean')
    recommendations['high_precision'] = {
        'use_case': 'Minimize False Positives (High Precision)',
        'description': 'Best for scenarios where false alarms are costly',
        'top_params': high_precision.iloc[0],
        'alternatives': high_precision.head(3)
    }
    
    # High Recall (minimize false negatives)
    high_recall = grouped_data.nlargest(10, 'recall_mean')
    recommendations['high_recall'] = {
        'use_case': 'Minimize False Negatives (High Recall)',
        'description': 'Best for scenarios where missing congestion events is costly',
        'top_params': high_recall.iloc[0],
        'alternatives': high_recall.head(3)
    }
    
    # Balanced Performance
    balanced = grouped_data.nlargest(10, 'f1_score_mean')
    recommendations['balanced'] = {
        'use_case': 'Balanced Performance (High F1-Score)',
        'description': 'Best overall trade-off between precision and recall',
        'top_params': balanced.iloc[0],
        'alternatives': balanced.head(3)
    }
    
    # Robust Performance
    robust = top_robust_combinations.head(10)
    recommendations['robust'] = {
        'use_case': 'Consistent Performance (High Robustness)',
        'description': 'Most consistent across different network conditions',
        'top_params': robust.iloc[0],
        'alternatives': robust.head(3)
    }
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))  # Increased figure size
    fig.suptitle('Heuristic Parameter Recommendations by Use Case', fontsize=18, fontweight='bold')
    
    use_cases = ['high_precision', 'high_recall', 'balanced', 'robust']
    titles = ['High Precision', 'High Recall', 'Balanced F1', 'Robust Performance']
    axes = [ax1, ax2, ax3, ax4]
    
    for ax, use_case, title in zip(axes, use_cases, titles):
        rec = recommendations[use_case]
        top_3 = rec['alternatives']
        
        # Create parameter combination strings
        param_strings = []
        scores = []
        for idx, row in top_3.iterrows():
            param_str = f"SW:{row['short_window']}\nLW:{row['long_window']}\nPT:{row['peak_tolerance']}\nCT:{row['count_threshold']}"
            param_strings.append(param_str)
            
            if use_case == 'high_precision':
                scores.append(row['precision_mean'])
            elif use_case == 'high_recall':
                scores.append(row['recall_mean'])
            elif use_case == 'balanced':
                scores.append(row['f1_score_mean'])
            else:  # robust
                scores.append(row['overall_robustness'])
        
        # Create bar plot
        bars = ax.bar(range(len(param_strings)), scores, color=['gold', 'silver', '#CD7F32'], 
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title(f'{title}\n{rec["description"]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(range(len(param_strings)))
        ax.set_xticklabels([f'#{i+1}' for i in range(len(param_strings))])
        ax.tick_params(labelsize=11)
        
        # Add parameter labels and scores - Fixed positioning
        for i, (bar, param_str, score) in enumerate(zip(bars, param_strings, scores)):
            height = bar.get_height()
            # Parameter labels above bars
            ax.text(i, height + 0.02, param_str, ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            # Score in middle of bars
            ax.text(i, height/2, f'{score:.3f}', ha='center', va='center', 
                   fontweight='bold', color='black', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('enhanced_parameter_analysis/7_heuristic_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return recommendations

heuristic_recommendations = generate_heuristic_recommendations()

# ===========================================
# SECTION 8: Peak Detection Visualization with Best Parameters
# ===========================================

def visualize_best_parameters():
    """Create peak detection visualization using the best parameters"""
    
    # Use the best balanced parameters
    best_params = best_combinations['best_balanced']
    
    print(f"\nUsing Best Balanced Parameters:")
    print(f"Short Window: {best_params['short_window']}ms")
    print(f"Long Window: {best_params['long_window']}ms") 
    print(f"Peak Tolerance: {best_params['peak_tolerance']}")
    print(f"Count Threshold: {best_params['count_threshold']}")
    print(f"Overall Score: {best_params['score']:.3f}")
    
    # Try to find a test case to visualize
    scenario_data = success_data[
        (success_data['file_name_clean'].str.contains('100mbps', case=False, na=False))
    ]
    
    if len(scenario_data) == 0:
        scenario_data = success_data.head(1)  # Fallback to any data
    
    test_case = scenario_data.iloc[0]
    
    # Load and process the test case
    raw_data_file = os.path.join('extracted_data', test_case['subfolder'], test_case['file_name'])
    
    try:
        raw_data = pd.read_csv(raw_data_file)
        
        # Calculate throughput using MATLAB-compatible method
        # For downstream, use tx_bytes; for upstream, use rx_bytes
        is_upstream = 'upstream' in test_case['file_name']
        byte_col = 'rx_bytes' if is_upstream else 'tx_bytes'
        fallback_col = 'tx_bytes' if is_upstream else 'rx_bytes'
        
        throughput, time = calculate_throughput(raw_data, byte_col, fallback_col)
        
        print(f"✅ Using MATLAB-compatible throughput calculation")
        print(f"   Direction: {'upstream' if is_upstream else 'downstream'}")
        print(f"   Primary byte column: {byte_col}")
        print(f"   Throughput range: {throughput.min():.2f} - {throughput.max():.2f} Mbps")
        
        if len(throughput) > 100 and not np.all(throughput == 0):
            # Run peak detection with best parameters
            clipping_binary, long_maxima, filtered_throughput, peak_ratio, short_counts, short_peaks = peak_speed_detect(
                throughput, time, 
                long_window_ms=best_params['long_window'], 
                short_window_ms=best_params['short_window'],
                peak_tolerance=best_params['peak_tolerance'], 
                count_threshold=best_params['count_threshold']
            )
            
            # Create visualization
            plt.figure(figsize=(20, 12))  # Increased figure size
            
            # Plot throughput
            plt.plot(time, throughput, label='Throughput (MATLAB method)', color='red', alpha=0.8, linewidth=2)
            
            # Add network capacity line
            network_capacity = test_case.get('network_limit', 100) * 1.15
            plt.axhline(y=network_capacity, color='black', linestyle='--', 
                       label=f'Network Capacity ({network_capacity:.0f} Mbps)', linewidth=3)
            
            # Add peak detection visualization
            mean_dt = np.mean(np.diff(time))
            long_window_samples = max(1, int(best_params['long_window'] / 1000 / mean_dt))
            short_window_samples = max(1, int(best_params['short_window'] / 1000 / mean_dt))
            
            # Plot long windows and clipping regions
            for long_start in range(0, len(throughput), long_window_samples):
                long_end = min(long_start + long_window_samples, len(throughput))
                long_max = long_maxima[long_start]
                
                if long_max > 0:
                    # Long window background
                    plt.fill_between(time[long_start:long_end], 0, long_max, 
                                   color='lightblue', alpha=0.3, step='post',
                                   label='Long Window' if long_start == 0 else "")
                    
                    # Check for clipping in short bins
                    for short_start in range(long_start, long_end, short_window_samples):
                        short_end = min(short_start + short_window_samples, long_end)
                        if short_start < len(clipping_binary) and clipping_binary[short_start] == 1:
                            count = short_counts[short_start]
                            
                            # Clipping region
                            plt.fill_between(time[short_start:short_end], 0, long_max, 
                                           color='orange', alpha=0.6, step='post',
                                           label='Clipping Detected' if long_start == 0 and short_start == long_start else "")
                            
                            # Add count label
                            mid_time = (time[short_start] + time[short_end-1]) / 2
                            plt.text(mid_time, long_max * 1.05, f'N={int(count)}', 
                                   ha='center', fontsize=12, fontweight='bold',
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))
            
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Throughput (Mbps)', fontsize=16)
            plt.title(f'Peak Detection with Optimized Parameters (MATLAB-Compatible Throughput)\n' +
                     f'File: {test_case["file_name"]} | ' +
                     f'SW:{best_params["short_window"]}ms, LW:{best_params["long_window"]}ms, ' +
                     f'PT:{best_params["peak_tolerance"]}, CT:{best_params["count_threshold"]}\n' +
                     f'Balanced Score: {best_params["score"]:.3f} | ' +
                     f'Acc:{best_params["accuracy"]:.3f}, Prec:{best_params["precision"]:.3f}, ' +
                     f'Rec:{best_params["recall"]:.3f}, F1:{best_params["f1_score"]:.3f}',
                     fontsize=16, fontweight='bold')
            plt.legend(loc='upper right', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tick_params(labelsize=14)
            plt.tight_layout()
            plt.savefig('enhanced_parameter_analysis/8_optimized_peak_detection.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Peak detection visualization saved using best parameters.")
            print(f"Clipping regions detected: {np.sum(clipping_binary)} samples")
            
    except Exception as e:
        print(f"Could not create peak detection visualization: {str(e)}")

visualize_best_parameters()

# ===========================================
# SECTION 9: Summary Report Generation
# ===========================================

def generate_summary_report():
    """Generate comprehensive summary report"""
    
    report = []
    report.append("="*80)
    report.append("ENHANCED PARAMETER ANALYSIS SUMMARY REPORT")
    report.append("(Using MATLAB-Compatible Throughput Calculation)")
    report.append("="*80)
    report.append(f"Total parameter combinations tested: {len(grouped_data)}")
    report.append(f"Total test cases: {len(success_data)}")
    report.append("")
    
    # Best combinations by metric
    report.append("BEST PARAMETER COMBINATIONS BY METRIC:")
    report.append("-"*50)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        best = best_combinations[f'best_{metric}']
        report.append(f"{best['name']}:")
        report.append(f"  Parameters: SW={best['short_window']}ms, LW={best['long_window']}ms, PT={best['peak_tolerance']}, CT={best['count_threshold']}")
        report.append(f"  Score: {best['score']:.4f} | Tests: {best['test_count']}")
        report.append("")
    
    # Best balanced combination
    best_bal = best_combinations['best_balanced']
    report.append("BEST BALANCED COMBINATION:")
    report.append("-"*30)
    report.append(f"Parameters: SW={best_bal['short_window']}ms, LW={best_bal['long_window']}ms, PT={best_bal['peak_tolerance']}, CT={best_bal['count_threshold']}")
    report.append(f"Balanced Score: {best_bal['score']:.4f}")
    report.append(f"Individual Scores - Acc:{best_bal['accuracy']:.3f}, Prec:{best_bal['precision']:.3f}, Rec:{best_bal['recall']:.3f}, F1:{best_bal['f1_score']:.3f}")
    report.append(f"Test Count: {best_bal['test_count']}")
    report.append("")
    
    # Heuristic recommendations
    report.append("HEURISTIC RECOMMENDATIONS BY USE CASE:")
    report.append("-"*50)
    for use_case, rec in heuristic_recommendations.items():
        top = rec['top_params']
        report.append(f"{rec['use_case']}:")
        report.append(f"  {rec['description']}")
        report.append(f"  Recommended: SW={top['short_window']}ms, LW={top['long_window']}ms, PT={top['peak_tolerance']}, CT={top['count_threshold']}")
        if use_case == 'high_precision':
            report.append(f"  Precision: {top['precision_mean']:.4f}")
        elif use_case == 'high_recall':
            report.append(f"  Recall: {top['recall_mean']:.4f}")
        elif use_case == 'balanced':
            report.append(f"  F1-Score: {top['f1_score_mean']:.4f}")
        else:
            report.append(f"  Robustness: {top['overall_robustness']:.4f}")
        report.append("")
    
    # Top robust combinations
    report.append("TOP 5 MOST ROBUST COMBINATIONS:")
    report.append("-"*40)
    for i, (idx, row) in enumerate(top_robust_combinations.head(5).iterrows()):
        report.append(f"{i+1}. SW={row['short_window']}ms, LW={row['long_window']}ms, PT={row['peak_tolerance']}, CT={row['count_threshold']}")
        report.append(f"   Robustness: {row['overall_robustness']:.4f} | F1: {row['f1_score_mean']:.3f} | Tests: {row['test_count']}")
        report.append("")
    
    # Parameter insights
    report.append("PARAMETER INSIGHTS:")
    report.append("-"*20)
    
    # Best ranges for each parameter
    for param, label in zip(['short_window', 'long_window', 'peak_tolerance', 'count_threshold'],
                           ['Short Window', 'Long Window', 'Peak Tolerance', 'Count Threshold']):
        best_f1_subset = grouped_data.nlargest(20, 'f1_score_mean')
        param_range = f"{best_f1_subset[param].min()}-{best_f1_subset[param].max()}"
        param_mode = best_f1_subset[param].mode().iloc[0] if len(best_f1_subset[param].mode()) > 0 else "N/A"
        report.append(f"{label}: Range in top performers: {param_range}, Most common: {param_mode}")
    
    report.append("")
    report.append("THROUGHPUT CALCULATION:")
    report.append("-"*25)
    report.append("✅ Using MATLAB-compatible method for consistency with feature extraction")
    report.append("   - Treats byte values as rates per interval (not cumulative)")
    report.append("   - Skips first row like MATLAB implementation")
    report.append("   - Better handling of counter resets and edge cases")
    report.append("")
    
    report.append("FILES GENERATED:")
    report.append("-"*15)
    report.append("1. 1_metric_distributions.png - Distribution of all metrics")
    report.append("2. 2_*_heatmap_*.png - Parameter pair heatmaps for all metrics")
    report.append("3. 3_parameter_importance.png - Parameter importance analysis")
    report.append("4. 4_best_combinations.png - Best parameter combinations")
    report.append("5. 5_parameter_sensitivity.png - Parameter sensitivity analysis")
    report.append("6. 6_robust_analysis.png - Robustness analysis")
    report.append("7. 7_heuristic_recommendations.png - Use case recommendations")
    report.append("8. 8_optimized_peak_detection.png - Peak detection with best parameters")
    report.append("")
    report.append("="*80)
    
    # Save report
    with open('enhanced_parameter_analysis/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS (with MATLAB-Compatible Throughput):")
    print("="*80)
    for line in report:
        if any(keyword in line for keyword in ['BEST BALANCED', 'Recommended:', 'Parameters:', 'Score:']):
            print(line)

generate_summary_report()

# Save detailed results to CSV
print("\nSaving detailed results...")
grouped_data.to_csv('enhanced_parameter_analysis/detailed_parameter_results.csv', index=False)
success_data.to_csv('enhanced_parameter_analysis/all_test_results.csv', index=False)

print("\n" + "="*80)
print("✅ ENHANCED PARAMETER ANALYSIS COMPLETE!")
print("="*80)
print("🔧 Now using MATLAB-compatible throughput calculation for consistency")
print("📊 Check the 'enhanced_parameter_analysis' folder for all results")
print(f"📁 Generated {len([f for f in os.listdir('enhanced_parameter_analysis') if os.path.isfile(os.path.join('enhanced_parameter_analysis', f))])} files with comprehensive multi-metric analysis")
print("🎯 Results should now be consistent with feature extraction pipeline")