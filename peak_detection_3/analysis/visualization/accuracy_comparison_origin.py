import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, roc_auc_score)

# Set global matplotlib parameters for IEEE paper
plt.rcParams.update({
    'font.size': 16,              # Increased base font size from 10 to 14
    'axes.titlesize': 16,         # Increased title font size from 11 to 16
    'axes.labelsize': 16,         # Increased axis label font size from 10 to 14
    'xtick.labelsize': 16,        # Increased X-axis tick label size from 9 to 12
    'ytick.labelsize': 16,        # Increased Y-axis tick label size from 9 to 12
    'legend.fontsize': 16,        # Increased legend font size from 9 to 12
    'figure.titlesize': 18,       # Increased figure title size from 12 to 18
    'figure.dpi': 300,            # Figure DPI for print quality (unchanged)
    'savefig.dpi': 300,           # Save figure DPI (unchanged)
    'savefig.bbox': 'tight',      # Tight bounding box when saving (unchanged)
    'savefig.pad_inches': 0.02,   # Minimal padding (unchanged)
    'lines.linewidth': 1.5,       # Line width (unchanged)
    'lines.markersize': 6,        # Marker size (unchanged)
    'axes.linewidth': 0.8,        # Axis line width (unchanged)
    'grid.linewidth': 0.8,        # Grid line width (unchanged)
    'figure.autolayout': True,    # Auto layout (unchanged)
})

# Define the LSTM model (unchanged)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.sigmoid(self.fc2(out))

# Helper functions
def extract_network_limit(subfolder):
    match = re.search(r'(\d+)$', subfolder)
    return int(match.group(1)) if match else -1

def extract_scenario_name(filename):
    """
    Extract scenario name in format: {speed}mbps_{load_type_combination}
    Handles filenames like:
    50mbps_fhd_limit_s2-mgmt_s2_downstream.csv → 50mbps_fhd_limit
    """
    import re

    # Remove .csv extension and test_ prefix if present
    base_name = filename.replace('.csv', '').replace('test_', '')
    
    # Remove the suffix starting from _s2 (like _s2-mgmt_s2_downstream)
    # This regex finds _s2 followed by anything until the end
    base_name = re.sub(r'_s2.*$', '', base_name)
    
    # Also remove standalone _downstream or _upstream at the end
    base_name = re.sub(r'_(downstream|upstream)$', '', base_name)
    
    return base_name

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

def preprocess_data(df, include_filename_features=False, include_direction=False, 
                   include_network_limit=True, time_steps=10, scaler=None, train_mode=False):
    data_copy = df.copy()
    
    # Replace infinities with NaN then fill with max value
    for col in data_copy.columns:
        if data_copy[col].dtype.kind in 'if':  # Only process numeric columns
            data_copy[col] = data_copy[col].replace([float('inf'), float('-inf')], float('nan'))
    
    data_copy = data_copy.fillna(data_copy.max())
    
    drop_cols = ['queue_size']
    if not include_filename_features:
        drop_cols.extend(['file_name', 'file_name_encoded'])
    else:
        drop_cols.append('file_name')
    if not include_direction and 'direction' in data_copy.columns:
        drop_cols.append('direction')
    if not include_network_limit and 'network_limit' in data_copy.columns:
        drop_cols.append('network_limit')
    
    data_copy = data_copy.drop(columns=[col for col in drop_cols if col in data_copy.columns])
    
    for col in ['subfolder', 'testcase_with_limit']:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
    
    feature_cols = [c for c in data_copy.columns if c != 'queue_exists' and c != 'detection_accuracy' and c != "time" and 
                    c != "relative_time" and c != 'network_limit']
    
    X = data_copy[feature_cols].values
    y = data_copy['queue_exists'].values
    
    if scaler is None and train_mode:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        temp_scaler = StandardScaler()
        X_scaled = temp_scaler.fit_transform(X)
    
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    return X_seq, y_seq, feature_cols, scaler

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    time_steps = checkpoint.get('time_steps', 10)
    return state_dict, time_steps

def evaluate_model(model, X, y, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Ensure inputs are PyTorch tensors
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.FloatTensor(X).to(device)
    else:
        X_tensor = X.to(device)
        
    if not isinstance(y, torch.Tensor):
        y_tensor = torch.FloatTensor(y).to(device)
    else:
        y_tensor = y.to(device)
    
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor), 
        batch_size=batch_size, shuffle=False
    )
    
    all_preds = []
    all_targets = []
    all_probs = []  # Store probability scores for ROC curve
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            
            # Store predictions, targets, and probabilities
            all_preds.extend(predicted.cpu().flatten().tolist())
            all_targets.extend(y_batch.cpu().flatten().tolist())
            all_probs.extend(outputs.cpu().flatten().tolist())  # Raw probability scores
    
    # Convert lists to PyTorch tensors for metrics calculation
    all_preds_tensor = torch.tensor(all_preds)
    all_targets_tensor = torch.tensor(all_targets)
    
    # Calculate metrics using PyTorch operations
    correct = (all_preds_tensor == all_targets_tensor).sum().item()
    total = len(all_targets_tensor)
    acc = correct / total if total > 0 else 0
    
    # For other metrics, convert to lists for sklearn functions
    all_preds_list = all_preds
    all_targets_list = all_targets
    all_probs_list = all_probs
    
    prec = precision_score(all_targets_list, all_preds_list, zero_division=0)
    rec = recall_score(all_targets_list, all_preds_list, zero_division=0)
    f1 = f1_score(all_targets_list, all_preds_list, zero_division=0)
    cm = confusion_matrix(all_targets_list, all_preds_list)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(all_targets_list, all_probs_list)
    except ValueError:
        # Handle case where all labels are the same class
        roc_auc = 0.5
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds_list,
        'targets': all_targets_list,
        'probabilities': all_probs_list  # Include probabilities for ROC curve
    }

def get_traditional_best_accuracy(param_tests_file, testcase):
    try:
        param_df = pd.read_csv(param_tests_file)
        print(f"  Loaded {len(param_df)} rows from {param_tests_file}")
        
        # Parse testcase (e.g., "50mbps_mixed_s2-mgmt_s2_100")
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        print(f"  Looking for testcase: {testcase}")
        print(f"  Parsed filename: '{filename}'")
        print(f"  Parsed network_limit: {network_limit}")
        
        # Look for downstream data with matching filename patterns
        expected_downstream = f"{filename}_downstream.csv"
        
        # Try exact match first
        case_params = param_df[
            (param_df['file_name'] == expected_downstream) & 
            (param_df['network_limit'] == network_limit) &
            (param_df['direction'] == 'downstream')
        ]
        
        # If no exact match, try partial filename matching for downstream
        if case_params.empty:
            case_params = param_df[
                (param_df['file_name'].str.contains(filename, na=False, case=False)) & 
                (param_df['network_limit'] == network_limit) &
                (param_df['direction'] == 'downstream')
            ]
        
        print(f"  Found {len(case_params)} total matching rows for {testcase}")
        
        if case_params.empty:
            print(f"  No parameter test data found for {testcase}")
            print(f"    Expected downstream filename: {expected_downstream}")
            
            # Debug: Show available downstream data
            downstream_data = param_df[param_df['direction'] == 'downstream']
            if not downstream_data.empty:
                print(f"    Available downstream files: {sorted(downstream_data['file_name'].unique())}")
                print(f"    Available network limits: {sorted(downstream_data['network_limit'].unique())}")
            else:
                print(f"    No downstream data found in parameter file")
            return None
        
        # Filter only successful tests (not skipped)
        successful_tests = case_params[case_params['status'] == 'success']
        print(f"  Found {len(successful_tests)} successful tests")
        
        if successful_tests.empty:
            print(f"  No successful tests found for {testcase}")
            print(f"  Available statuses: {case_params['status'].unique()}")
            return None
        
        # Show sample of successful tests
        print(successful_tests[['short_window', 'long_window', 'peak_tolerance', 'count_threshold', 'accuracy']].head())
        
        best_row = successful_tests.loc[successful_tests['accuracy'].idxmax()]
        print(f"  Best accuracy: {best_row['accuracy']:.4f}")
        
        return {
            'accuracy': best_row['accuracy'],
            'f1_score': best_row['f1_score'],
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'short_window': best_row['short_window'],
            'long_window': best_row['long_window'],
            'peak_tolerance': best_row['peak_tolerance'],
            'count_threshold': best_row['count_threshold']
        }
    except Exception as e:
        print(f"  Error getting traditional best accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_custom_heuristic_accuracy(param_tests_file, testcase, custom_params):
    """
    Get the accuracy for traditional method with custom parameters
    
    Parameters:
    - param_tests_file: Path to parameter test results CSV
    - testcase: The test case identifier
    - custom_params: Dict with keys: short_window, long_window, peak_tolerance, count_threshold
    """
    try:
        param_df = pd.read_csv(param_tests_file)
        
        # Parse testcase (e.g., "50mbps_mixed_s2-mgmt_s2_100")
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        # Look for downstream data with matching filename patterns
        expected_downstream = f"{filename}_downstream.csv"
        
        # Try exact match first
        case_params = param_df[
            (param_df['file_name'] == expected_downstream) & 
            (param_df['network_limit'] == network_limit) &
            (param_df['direction'] == 'downstream') &
            (param_df['status'] == 'success')
        ]
        
        # If no exact match, try partial filename matching for downstream
        if case_params.empty:
            case_params = param_df[
                (param_df['file_name'].str.contains(filename, na=False, case=False)) & 
                (param_df['network_limit'] == network_limit) &
                (param_df['direction'] == 'downstream') &
                (param_df['status'] == 'success')
            ]
        
        if case_params.empty:
            print(f"  No parameter test data found for {testcase}")
            return None
        
        # Check if exact parameter combination exists
        exact_match = case_params[
            (case_params['short_window'] == custom_params['short_window']) &
            (case_params['long_window'] == custom_params['long_window']) &
            (case_params['peak_tolerance'] == custom_params['peak_tolerance']) &
            (case_params['count_threshold'] == custom_params['count_threshold'])
        ]
        
        if not exact_match.empty:
            # Found exact match
            result_row = exact_match.iloc[0]
            print(f"  Found exact parameter match for {testcase}")
        else:
            # Find closest match
            print(f"  No exact parameter match found for {testcase}, looking for closest match...")
            print(f"  Looking for: sw={custom_params['short_window']}, lw={custom_params['long_window']}, pt={custom_params['peak_tolerance']}, ct={custom_params['count_threshold']}")
            
            # Show available parameter ranges
            print(f"  Available short_window range: {case_params['short_window'].min()}-{case_params['short_window'].max()}")
            print(f"  Available long_window range: {case_params['long_window'].min()}-{case_params['long_window'].max()}")
            print(f"  Available peak_tolerance range: {case_params['peak_tolerance'].min():.3f}-{case_params['peak_tolerance'].max():.3f}")
            print(f"  Available count_threshold range: {case_params['count_threshold'].min()}-{case_params['count_threshold'].max()}")
            
            # Calculate distance for each parameter combination
            case_params = case_params.copy()
            case_params['param_distance'] = (
                abs(case_params['short_window'] - custom_params['short_window']) +
                abs(case_params['long_window'] - custom_params['long_window']) +
                abs(case_params['peak_tolerance'] - custom_params['peak_tolerance']) * 100 +  # Scale up tolerance
                abs(case_params['count_threshold'] - custom_params['count_threshold'])
            )
            
            # Find the closest match
            closest_match = case_params.loc[case_params['param_distance'].idxmin()]
            
            print(f"  Closest match: sw={closest_match['short_window']}, lw={closest_match['long_window']}, pt={closest_match['peak_tolerance']:.3f}, ct={closest_match['count_threshold']}")
            print(f"  Distance: {closest_match['param_distance']:.2f}")
            
            result_row = closest_match
        
        return {
            'accuracy': result_row['accuracy'],
            'f1_score': result_row['f1_score'],
            'precision': result_row['precision'],
            'recall': result_row['recall'],
            'short_window': result_row['short_window'],
            'long_window': result_row['long_window'],
            'peak_tolerance': result_row['peak_tolerance'],
            'count_threshold': result_row['count_threshold']
        }
    except Exception as e:
        print(f"  Error getting custom heuristic accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_roc_curve_plot(comparison_results, output_dir, model_types=['finetune', 'combined']):
    """
    Create ROC curve plots for different models
    """
    print(f"\nGenerating ROC curve plots...")
    
    # Create individual ROC curves for each test case
    n_testcases = len(comparison_results)
    n_cols = min(3, n_testcases)
    n_rows = (n_testcases + n_cols - 1) // n_cols
    
    if n_testcases > 1:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_testcases > 1 else [axes]
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    
    # Collect all ROC data for aggregate plot
    all_fpr = {}
    all_tpr = {}
    all_aucs = {}
    
    for model_type in model_types:
        all_fpr[model_type] = []
        all_tpr[model_type] = []
        all_aucs[model_type] = []
    
    for idx, result in enumerate(comparison_results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        scenario = result['scenario']
        
        # Plot ROC curves for each model type
        for model_type in model_types:
            if model_type == 'finetune':
                model_data = result['finetune_model']
                color = 'green'
                label = 'Finetune Model'
            elif model_type == 'combined' and 'combined_model' in result and result['combined_model'].get('exists', True):
                model_data = result['combined_model']
                color = 'purple'
                label = 'Combined Model'
            else:
                continue
            
            if 'metrics' in model_data:
                targets = model_data['metrics']['targets']
                probabilities = model_data['metrics']['probabilities']
                roc_auc = model_data['metrics']['roc_auc']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(targets, probabilities)
                
                # Store for aggregate plot
                all_fpr[model_type].append(fpr)
                all_tpr[model_type].append(tpr)
                all_aucs[model_type].append(roc_auc)
                
                # Plot individual ROC curve
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{label} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {scenario}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(comparison_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_individual.png'), dpi=300, bbox_inches='tight')
    print(f"Individual ROC curves saved to {os.path.join(output_dir, 'roc_curves_individual.png')}")
    plt.close()
    
    # Create aggregate ROC curve plot
    plt.figure(figsize=(10, 8))
    
    for model_type in model_types:
        if not all_aucs[model_type]:  # Skip if no data
            continue
            
        # Calculate mean ROC curve using interpolation
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = all_aucs[model_type]
        
        for fpr, tpr in zip(all_fpr[model_type], all_tpr[model_type]):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            # Confidence interval
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            
            if model_type == 'finetune':
                color = 'green'
                label = f'Finetune Model (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})'
            elif model_type == 'combined':
                color = 'purple'
                label = f'Combined Model (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})'
            
            plt.plot(mean_fpr, mean_tpr, color=color, linewidth=3, label=label)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Mean ROC Curves with Confidence Intervals', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'roc_curves_aggregate.png'), dpi=300, bbox_inches='tight')
    print(f"Aggregate ROC curves saved to {os.path.join(output_dir, 'roc_curves_aggregate.png')}")
    plt.close()

def create_simple_line_comparison(comparison_results, output_dir, metric='accuracy', has_combined=True):
    """
    Create a simple line chart comparing models across test cases
    """
    print(f"\nGenerating simple {metric} line comparison...")
    
    # Extract data for plotting
    testcase_labels = [r['scenario'] for r in comparison_results]
    finetune_values = [float(r['finetune_model']['metrics'][metric]) for r in comparison_results]
    
    plt.figure(figsize=(16, 6))
    positions = range(len(testcase_labels))
    
    # Plot finetune model
    plt.plot(positions, finetune_values, 'g-o', label='Finetune Model', linewidth=2, markersize=8)
    
    # Plot combined model if available
    if has_combined:
        combined_values = [float(r['combined_model']['metrics'][metric]) for r in comparison_results]
        plt.plot(positions, combined_values, 'purple', linestyle='--', marker='o', label='Combined Model', linewidth=2, markersize=8)
    
    # Customize plot
    plt.xticks(positions, testcase_labels, rotation=45, ha='right')
    plt.xlabel('Test Case')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Comparison: Finetune vs Combined Model')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on points
    for i, (pos, val) in enumerate(zip(positions, finetune_values)):
        plt.annotate(f'{val:.2f}', (pos, val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')
    
    if has_combined:
        for i, (pos, val) in enumerate(zip(positions, combined_values)):
            plt.annotate(f'{val:.2f}', (pos, val), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='purple')
    
    plt.tight_layout()
    
    plot_filename = f'{metric}_line_comparison.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    print(f"Simple line comparison saved to {os.path.join(output_dir, plot_filename)}")
    plt.close()

def create_auc_bar_chart(comparison_results, output_dir, has_combined=True):
    """
    Create a bar chart comparing AUC scores across models and test cases
    """
    print(f"\nGenerating AUC bar chart...")
    
    scenarios = [r['scenario'] for r in comparison_results]
    finetune_aucs = [float(r['finetune_model']['metrics']['roc_auc']) for r in comparison_results]
    
    print(f"Creating AUC bar chart for {len(scenarios)} scenarios")
    print(f"Finetune AUCs: {[f'{auc:.2f}' for auc in finetune_aucs]}")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(scenarios))
    width = 0.35 if has_combined else 0.5
    
    # Create bars
    bars1 = ax.bar(x - width/2 if has_combined else x, finetune_aucs, width, 
                   label='Finetune Model', color='green', alpha=0.8)
    
    if has_combined:
        combined_aucs = [float(r['combined_model']['metrics']['roc_auc']) for r in comparison_results]
        print(f"Combined AUCs: {[f'{auc:.2f}' for auc in combined_aucs]}")
        bars2 = ax.bar(x + width/2, combined_aucs, width, 
                       label='Combined Model', color='purple', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Test Scenarios', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC Performance Comparison Across Models and Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    if has_combined:
        add_labels(bars2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'auc_bar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AUC bar chart saved to {output_path}")
    plt.close()

def create_auc_heatmap(comparison_results, output_dir, has_combined=True):
    """
    Create a heatmap showing AUC performance across scenarios and models
    """
    print(f"\nGenerating AUC heatmap...")
    
    scenarios = [r['scenario'] for r in comparison_results]
    print(f"Creating heatmap for {len(scenarios)} scenarios")
    
    # Prepare data matrix
    models = ['Finetune']
    if has_combined:
        models.append('Combined')
    
    auc_matrix = []
    for model in models:
        model_aucs = []
        for result in comparison_results:
            if model == 'Finetune':
                auc = float(result['finetune_model']['metrics']['roc_auc'])
            elif model == 'Combined':
                auc = float(result['combined_model']['metrics']['roc_auc'])
            model_aucs.append(auc)
        auc_matrix.append(model_aucs)
        print(f"{model} AUCs: {[f'{auc:.2f}' for auc in model_aucs]}")
    
    auc_matrix = np.array(auc_matrix)
    print(f"AUC matrix shape: {auc_matrix.shape}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(scenarios)), 6))
    im = ax.imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{auc_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title("AUC Performance Heatmap", fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Scenarios', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'auc_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AUC heatmap saved to {output_path}")
    plt.close()

def create_auc_distribution_plot(comparison_results, output_dir, has_combined=True):
    """
    Create distribution plots showing AUC score distributions
    """
    print(f"\nGenerating AUC distribution plot...")
    
    finetune_aucs = [float(r['finetune_model']['metrics']['roc_auc']) for r in comparison_results]
    print(f"Finetune AUC stats: mean={np.mean(finetune_aucs):.2f}, std={np.std(finetune_aucs):.2f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    bins = np.linspace(0.5, 1.0, 20)
    ax1.hist(finetune_aucs, bins=bins, alpha=0.7, color='green', label='Finetune Model', density=True)
    
    if has_combined:
        combined_aucs = [float(r['combined_model']['metrics']['roc_auc']) for r in comparison_results]
        print(f"Combined AUC stats: mean={np.mean(combined_aucs):.2f}, std={np.std(combined_aucs):.2f}")
        ax1.hist(combined_aucs, bins=bins, alpha=0.7, color='purple', label='Combined Model', density=True)
    
    ax1.set_xlabel('AUC Score')
    ax1.set_ylabel('Density')
    ax1.set_title('AUC Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [finetune_aucs]
    labels = ['Finetune']
    
    if has_combined:
        data_to_plot.append(combined_aucs)
        labels.append('Combined')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    colors = ['green', 'purple']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC Score Box Plot')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.5, 1.0)
    
    # Add statistics text
    stats_text = f"Finetune: μ={np.mean(finetune_aucs):.2f}, σ={np.std(finetune_aucs):.2f}"
    if has_combined:
        stats_text += f"\nCombined: μ={np.mean(combined_aucs):.2f}, σ={np.std(combined_aucs):.2f}"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'auc_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AUC distribution plot saved to {output_path}")
    plt.close()

def create_performance_radar_chart(comparison_results, output_dir, has_combined=True):
    """
    Create radar chart comparing multiple metrics including AUC
    """
    print(f"\nGenerating performance radar chart...")
    
    # Calculate average metrics across all test cases
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    # Calculate averages for finetune model
    finetune_avgs = []
    for metric in metrics:
        values = [float(r['finetune_model']['metrics'][metric]) for r in comparison_results]
        avg_val = np.mean(values)
        finetune_avgs.append(avg_val)
        print(f"Finetune {metric}: {avg_val:.2f}")
    
    # Calculate averages for combined model if available
    if has_combined:
        combined_avgs = []
        for metric in metrics:
            values = [float(r['combined_model']['metrics'][metric]) for r in comparison_results]
            avg_val = np.mean(values)
            combined_avgs.append(avg_val)
            print(f"Combined {metric}: {avg_val:.2f}")
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot finetune model
    finetune_values = finetune_avgs + [finetune_avgs[0]]  # Close the polygon
    ax.plot(angles, finetune_values, 'o-', linewidth=2, label='Finetune Model', color='green')
    ax.fill(angles, finetune_values, alpha=0.25, color='green')
    
    # Plot combined model if available
    if has_combined:
        combined_values = combined_avgs + [combined_avgs[0]]  # Close the polygon
        ax.plot(angles, combined_values, 'o-', linewidth=2, label='Combined Model', color='purple')
        ax.fill(angles, combined_values, alpha=0.25, color='purple')
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart\n(Average Across All Test Cases)', 
                size=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Add value annotations
    for angle, value, label in zip(angles[:-1], finetune_values[:-1], metric_labels):
        ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='center', 
                fontsize=9, color='green', fontweight='bold')
    
    if has_combined:
        for angle, value, label in zip(angles[:-1], combined_values[:-1], metric_labels):
            ax.text(angle, value - 0.05, f'{value:.2f}', ha='center', va='center', 
                    fontsize=9, color='purple', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance radar chart saved to {output_path}")
    plt.close()

def create_auc_performance_table(comparison_results, output_dir, has_combined=True):
    """
    Create a visual performance table with AUC rankings
    """
    print(f"\nGenerating AUC performance table...")
    
    # Prepare data
    table_data = []
    for result in comparison_results:
        row = {
            'Scenario': result['scenario'],
            'Sample Count': result['sample_count'],
            'Finetune AUC': float(result['finetune_model']['metrics']['roc_auc']),
            'Finetune Acc': float(result['finetune_model']['metrics']['accuracy'])
        }
        
        if has_combined:
            row['Combined AUC'] = float(result['combined_model']['metrics']['roc_auc'])
            row['Combined Acc'] = float(result['combined_model']['metrics']['accuracy'])
            row['AUC Diff'] = row['Finetune AUC'] - row['Combined AUC']
        
        table_data.append(row)
    
    print(f"Creating table with {len(table_data)} rows")
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    if has_combined:
        columns = ['Scenario', 'Samples', 'Finetune\nAUC', 'Combined\nAUC', 'AUC\nDiff', 'Finetune\nAcc', 'Combined\nAcc']
        cell_text = []
        for _, row in df.iterrows():
            cell_text.append([
                row['Scenario'][:20] + '...' if len(row['Scenario']) > 20 else row['Scenario'],
                f"{row['Sample Count']:,}",
                f"{row['Finetune AUC']:.2f}",
                f"{row['Combined AUC']:.2f}",
                f"{row['AUC Diff']:+.2f}",
                f"{row['Finetune Acc']:.2f}",
                f"{row['Combined Acc']:.2f}"
            ])
    else:
        columns = ['Scenario', 'Samples', 'Finetune\nAUC', 'Finetune\nAcc']
        cell_text = []
        for _, row in df.iterrows():
            cell_text.append([
                row['Scenario'][:20] + '...' if len(row['Scenario']) > 20 else row['Scenario'],
                f"{row['Sample Count']:,}",
                f"{row['Finetune AUC']:.2f}",
                f"{row['Finetune Acc']:.2f}"
            ])
    
    # Color cells based on performance
    cell_colors = []
    for i, row in df.iterrows():
        row_colors = ['lightgray'] * len(columns)  # Default color
        
        # Color AUC cells based on performance
        auc_col_idx = 2  # Finetune AUC column
        if row['Finetune AUC'] >= 0.9:
            row_colors[auc_col_idx] = 'lightgreen'
        elif row['Finetune AUC'] >= 0.8:
            row_colors[auc_col_idx] = 'lightyellow'
        else:
            row_colors[auc_col_idx] = 'lightcoral'
        
        if has_combined:
            combined_auc_col_idx = 3  # Combined AUC column
            if row['Combined AUC'] >= 0.9:
                row_colors[combined_auc_col_idx] = 'lightgreen'
            elif row['Combined AUC'] >= 0.8:
                row_colors[combined_auc_col_idx] = 'lightyellow'
            else:
                row_colors[combined_auc_col_idx] = 'lightcoral'
            
            # Color difference column
            diff_col_idx = 4
            if row['AUC Diff'] > 0.01:
                row_colors[diff_col_idx] = 'lightgreen'
            elif row['AUC Diff'] < -0.01:
                row_colors[diff_col_idx] = 'lightcoral'
            else:
                row_colors[diff_col_idx] = 'lightyellow'
        
        cell_colors.append(row_colors)
    
    table = ax.table(cellText=cell_text, colLabels=columns, cellColors=cell_colors,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('AUC Performance Summary Table\n(Green: >0.9, Yellow: 0.8-0.9, Red: <0.8)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'auc_performance_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AUC performance table saved to {output_path}")
    plt.close()

def create_comparison_plot(comparison_results, param_df, output_dir, metric='accuracy', 
                          has_combined=True, has_traditional=True, custom_heuristic_params=None):
    """
    Create a comparison plot for a specific metric (accuracy, f1_score, or roc_auc)
    """
    print(f"\nGenerating {metric} comparison plot...")
    
    # Extract data for plotting
    testcase_labels = [r['scenario'] for r in comparison_results]
    finetune_values = [float(r['finetune_model']['metrics'][metric]) for r in comparison_results]
    
    # Add combined model values if available
    if has_combined:
        combined_values = [float(r['combined_model']['metrics'][metric]) for r in comparison_results]
    
    # Add custom heuristic values if parameters provided (only for accuracy, f1_score)
    custom_heuristic_values = []
    has_custom_heuristic = False
    if custom_heuristic_params and has_traditional and metric in ['accuracy', 'f1_score']:
        print(f"Evaluating custom heuristic with parameters: {custom_heuristic_params}")
        has_custom_heuristic = True
        
        for result in comparison_results:
            testcase = result['testcase']
            param_tests_file = 'combined_param_tests.csv'  # Adjust path as needed
            custom_result = get_custom_heuristic_accuracy(param_tests_file, testcase, custom_heuristic_params)
            
            if custom_result:
                custom_heuristic_values.append(float(custom_result[metric]))
                print(f"  Custom heuristic {metric} for {testcase}: {custom_result[metric]:.4f}")
            else:
                custom_heuristic_values.append(0.0)  # Default value if no result
                print(f"  Custom heuristic {metric} for {testcase}: No data")
    
    if not has_traditional or metric == 'roc_auc':
        print(f"No traditional data available for {metric} plotting.")
        
        # If no traditional data, just plot finetune vs combined
        if has_combined:
            plt.figure(figsize=(16, 6))
            positions = range(len(testcase_labels))
            
            plt.plot(positions, finetune_values, 'g-o', label='Finetune Model', linewidth=2, markersize=8)
            plt.plot(positions, combined_values, 'purple', linestyle='--', marker='o', label='Combined Model', linewidth=2, markersize=8)
            
            plt.xticks(positions, testcase_labels, rotation=45, ha='right')
            plt.xlabel('Test Case')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Comparison: Finetune vs Combined Model')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_filename = f'{metric}_comparison_plot.png'
            plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
            print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
            plt.close()
        
        return

    # Get all traditional values for each test case (for boxplots) - only for accuracy and f1_score
    traditional_values_per_testcase = []
    debug_info = []

    for result in comparison_results:
        testcase = result['testcase']
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        print(f"\nLooking for {metric} boxplot data for testcase: {testcase}")
        print(f"  Parsed filename: '{filename}'")
        print(f"  Parsed network_limit: {network_limit}")
        
        # Look for downstream data with matching filename patterns
        expected_downstream = f"{filename}_downstream.csv"
        
        # Try exact match first
        case_params = param_df[
            (param_df['file_name'] == expected_downstream) & 
            (param_df['network_limit'] == network_limit) &
            (param_df['direction'] == 'downstream') &
            (param_df['status'] == 'success')  # Only successful tests
        ]
        
        # If no exact match, try partial filename matching for downstream
        if case_params.empty:
            case_params = param_df[
                (param_df['file_name'].str.contains(filename, na=False, case=False)) & 
                (param_df['network_limit'] == network_limit) &
                (param_df['direction'] == 'downstream') &
                (param_df['status'] == 'success')  # Only successful tests
            ]
        
        print(f"  Found {len(case_params)} successful downstream tests")
        
        if not case_params.empty:
            values = case_params[metric].values.tolist()
            traditional_values_per_testcase.append(values)
            debug_info.append(f"{testcase}: {len(values)} values")
            print(f"  {metric.title()} range: {min(values):.3f} - {max(values):.3f}")
        else:
            traditional_values_per_testcase.append([])  # Empty list if no data
            debug_info.append(f"{testcase}: NO DATA")
            print(f"  No successful downstream data found")

    print(f"\n{metric.title()} boxplot data summary:")
    for info in debug_info:
        print(f"  {info}")

    # Check if we have any data for boxplots
    total_data_points = sum(len(val_list) for val_list in traditional_values_per_testcase)
    print(f"\nTotal data points for {metric} boxplot: {total_data_points}")

    if total_data_points == 0:
        print(f"ERROR: No traditional {metric} data found for any test case!")
        
        # Still create plot without boxplots but with custom heuristic if available
        plt.figure(figsize=(16, 6))
        positions = range(len(testcase_labels))
        
        # Line plot for finetune model values
        plt.plot(positions, finetune_values, 'g--o', label='Finetune Model', linewidth=2, markersize=8)
        
        # Line plot for combined model values if available
        if has_combined:
            plt.plot(positions, combined_values, 'purple', linestyle='-.', marker='o', label='Combined Model', linewidth=2, markersize=8)
        
        # Line plot for custom heuristic if available
        if has_custom_heuristic:
            plt.plot(positions, custom_heuristic_values, 'r-s', 
                    label='Custom Heuristic', 
                    linewidth=2, markersize=8)

        # Customize plot
        plt.xticks(positions, testcase_labels, rotation=45, ha='right')
        plt.xlabel('Test Case')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison: LSTM Models Only (No Traditional Data Available)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save plot
        plot_filename = f'{metric}_comparison_plot.png'
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
        plt.close()
        return

    # Create figure and axis only if we have data
    plt.figure(figsize=(16, 6))

    # Box plots for traditional values per test case
    positions = range(len(testcase_labels))

    # Filter out empty lists for boxplot
    non_empty_positions = []
    non_empty_data = []
    non_empty_labels = []

    for i, (pos, data, label) in enumerate(zip(positions, traditional_values_per_testcase, testcase_labels)):
        if len(data) > 0:  # Only include if we have data
            non_empty_positions.append(pos)
            non_empty_data.append(data)
            non_empty_labels.append(label)

    if non_empty_data:
        print(f"Creating {metric} boxplots for {len(non_empty_data)} test cases with data")
        plt.boxplot(non_empty_data, positions=non_empty_positions, widths=0.6, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='blue'), 
                    whiskerprops=dict(color='blue'), capprops=dict(color='blue'), 
                    medianprops=dict(color='red'))

    # Line plot for finetune model values
    plt.plot(positions, finetune_values, 'g--o', label='Finetune Model', linewidth=2, markersize=8)

    # Line plot for combined model values if available
    if has_combined:
        plt.plot(positions, combined_values, 'purple', linestyle='-.', marker='o', label='Combined Model', linewidth=2, markersize=8)

    # Line plot for custom heuristic if available
    if has_custom_heuristic:
        plt.plot(positions, custom_heuristic_values, 'r-s', 
                label='Custom Heuristic', 
                linewidth=2, markersize=8)

    # Add legend entry for traditional method boxplots
    if non_empty_data:
        plt.plot([], [], 's', color='lightblue', label='Traditional (All Params)')

    # Customize plot
    plt.xticks(positions, testcase_labels, rotation=45, ha='right')
    plt.xlabel('Test Case')
    plt.ylabel(metric.replace('_', ' ').title())
    
    title_parts = [f'{metric.replace("_", " ").title()} Comparison:']
    if non_empty_data:
        title_parts.append('Traditional (Box)')
    if has_custom_heuristic:
        title_parts.append('Custom Heuristic (Red)')
    title_parts.extend(['Finetune (Green)'])
    if has_combined:
        title_parts.append('Combined (Purple)')
    
    plt.title(', '.join(title_parts))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plot_filename = f'{metric}_comparison_plot.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
    plt.close()

def main():
    # Updated directory structure
    results_dir = 'results'
    models_dir = os.path.join(results_dir, 'models')
    output_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # CUSTOM HEURISTIC PARAMETERS - MODIFY THESE AS NEEDED
    # These are more common parameter values that are likely to exist in the test file
    custom_heuristic_params = {
        'short_window': 200,       
        'long_window': 28500,      
        'peak_tolerance': 0.3,  
        'count_threshold': 3    
    }
    
    print("Loading dataset...")
    data = pd.read_csv('combined_features.csv')
    print(f"Loaded {len(data)} records")
    
    param_tests_file = os.path.abspath('combined_param_tests.csv')
    if not os.path.exists(param_tests_file):
        print(f"Warning: Traditional parameter test file not found at {param_tests_file}")
        has_traditional = False
    else:
        print(f"Loading traditional parameter test results from {param_tests_file}")
        param_df = pd.read_csv(param_tests_file)  # Load param tests for all accuracies
        has_traditional = True
        
        # Debug parameter file info
        print(f"Parameter test data info:")
        print(f"  Total rows: {len(param_df)}")
        print(f"  Downstream rows: {len(param_df[param_df['direction'] == 'downstream'])}")
        print(f"  Upstream rows: {len(param_df[param_df['direction'] == 'upstream'])}")
        print(f"  Unique directions: {sorted(param_df['direction'].unique())}")
        print(f"  Unique statuses: {sorted(param_df['status'].unique())}")
        successful_downstream = param_df[(param_df['direction'] == 'downstream') & (param_df['status'] == 'success')]
        print(f"  Successful downstream tests: {len(successful_downstream)}")
        if len(successful_downstream) > 0:
            print(f"  Downstream file examples: {sorted(successful_downstream['file_name'].unique())[:3]}")
    
    if 'network_limit' not in data.columns and 'subfolder' in data.columns:
        data['network_limit'] = data['subfolder'].apply(extract_network_limit)
    elif 'network_limit' not in data.columns:
        data['network_limit'] = -1
    
    downstream_data = data[data['direction'] == 'downstream'].copy()
    downstream_data['testcase_with_limit'] = downstream_data.apply(
        lambda row: f"{row['file_name']}_{row['network_limit']}", axis=1
    )
    
    filename_encoder = LabelEncoder()
    data['file_name_encoded'] = filename_encoder.fit_transform(data['file_name'])
    downstream_data['file_name_encoded'] = filename_encoder.transform(downstream_data['file_name'])
    
    unique_testcases = downstream_data['testcase_with_limit'].unique()
    print(f"Found {len(unique_testcases)} unique test case + network limit combinations")
    print(f"Example testcases: {list(unique_testcases[:5])}")
    
    # Load finetune model (looking for final model first)
    finetune_model_path = os.path.join(models_dir, 'finetune_final_model.pt')
    if not os.path.exists(finetune_model_path):
        # Try looking for the last finetune step model
        finetune_files = [f for f in os.listdir(models_dir) if f.startswith('finetune_step') and f.endswith('.pt')]
        if finetune_files:
            # Get the highest step number
            step_nums = [int(re.search(r'step(\d+)', f).group(1)) for f in finetune_files]
            highest_step = max(step_nums)
            finetune_model_path = os.path.join(models_dir, [f for f in finetune_files if f'step{highest_step}' in f][0])
        else:
            print("Error: No finetune model found")
            return

    print(f"Loading finetune model from {finetune_model_path}")
    finetune_state_dict, finetune_time_steps = load_model(finetune_model_path)
    
    # Load combined model
    combined_model_path = os.path.join(models_dir, 'combined_model_downstream_only.pt')
    if not os.path.exists(combined_model_path):
        print(f"Warning: Combined model not found at {combined_model_path}")
        has_combined = False
    else:
        print(f"Loading combined model from {combined_model_path}")
        combined_state_dict, combined_time_steps = load_model(combined_model_path)
        has_combined = True
    
    comparison_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for testcase in unique_testcases:
        print(f"\nEvaluating on testcase: {testcase}")
        
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        # Skip test cases that contain 'fhd' in the filename
        if 'fhd' in filename.lower():
            print(f"  Skipping {testcase} - contains 'fhd' in filename")
            continue
            
        scenario = extract_scenario_name(filename)
        print("Scenario: ", scenario)
        
        test_data = downstream_data[
            (downstream_data['file_name'] == filename) & 
            (downstream_data['network_limit'] == network_limit)
        ]
        
        if len(test_data) <= 15:
            print(f"  Skipping {testcase} - insufficient data ({len(test_data)} rows)")
            continue
        
        results_entry = {
            'testcase': testcase,
            'scenario': scenario,
            'network_limit': network_limit,
            'sample_count': int(len(test_data))
        }
        
        # Evaluate finetune model
        print(f"  Evaluating finetune model")
        X_test, y_test, feature_cols, _ = preprocess_data(
            test_data, include_network_limit=True, time_steps=finetune_time_steps
        )
        finetune_model = LSTMModel(input_size=X_test.shape[2])
        finetune_model.load_state_dict(finetune_state_dict)
        finetune_model.to(device)
        finetune_metrics = evaluate_model(finetune_model, X_test, y_test)
        results_entry['finetune_model'] = {'metrics': finetune_metrics}
        print(f"  Finetune model accuracy: {finetune_metrics['accuracy']:.2f}, AUC: {finetune_metrics['roc_auc']:.2f}")
        
        # Evaluate combined model if available
        if has_combined:
            print(f"  Evaluating combined model")
            X_test, y_test, feature_cols, _ = preprocess_data(
                test_data, include_network_limit=True, time_steps=combined_time_steps
            )
            combined_model = LSTMModel(input_size=X_test.shape[2])
            combined_model.load_state_dict(combined_state_dict)
            combined_model.to(device)
            combined_metrics = evaluate_model(combined_model, X_test, y_test)
            results_entry['combined_model'] = {'metrics': combined_metrics}
            print(f"  Combined model accuracy: {combined_metrics['accuracy']:.2f}, AUC: {combined_metrics['roc_auc']:.2f}")
        else:
            results_entry['combined_model'] = {'exists': False}
        
        # Get traditional accuracy if available
        if has_traditional:
            traditional_best = get_traditional_best_accuracy(param_tests_file, testcase)
            if traditional_best:
                results_entry['traditional_best'] = {'exists': True, 'metrics': traditional_best}
                print(f"  Traditional best accuracy: {traditional_best['accuracy']:.2f}")
            else:
                results_entry['traditional_best'] = {'exists': False}
        
        comparison_results.append(results_entry)
    
    print("\nSummarizing results...")
    summary_data = []
    for result in comparison_results:
        row = {
            'testcase': result['testcase'],
            'scenario': result['scenario'],
            'network_limit': result['network_limit'],
            'sample_count': result['sample_count'],
            'finetune_accuracy': float(result['finetune_model']['metrics']['accuracy']),
            'finetune_f1': float(result['finetune_model']['metrics']['f1_score']),
            'finetune_auc': float(result['finetune_model']['metrics']['roc_auc']),
        }
        
        # Add combined model metrics if available
        if has_combined:
            comb_metrics = result['combined_model']['metrics']
            row.update({
                'combined_accuracy': float(comb_metrics['accuracy']),
                'combined_f1': float(comb_metrics['f1_score']),
                'combined_auc': float(comb_metrics['roc_auc']),
                'finetune_vs_combined_accuracy': float(result['finetune_model']['metrics']['accuracy'] - comb_metrics['accuracy']),
                'finetune_vs_combined_auc': float(result['finetune_model']['metrics']['roc_auc'] - comb_metrics['roc_auc']),
            })
        
        # Add traditional metrics if available
        if has_traditional and result.get('traditional_best', {}).get('exists', False):
            trad_metrics = result['traditional_best']['metrics']
            row.update({
                'traditional_accuracy': float(trad_metrics['accuracy']),
                'traditional_f1': float(trad_metrics['f1_score']),
                'finetune_vs_traditional': float(result['finetune_model']['metrics']['accuracy'] - trad_metrics['accuracy']),
            })
            
            if has_combined:
                row['combined_vs_traditional'] = float(comb_metrics['accuracy'] - trad_metrics['accuracy'])
                accuracies = {
                    'Combined': float(comb_metrics['accuracy']),
                    'Finetune': float(result['finetune_model']['metrics']['accuracy']),
                    'Traditional': float(trad_metrics['accuracy'])
                }
                row['best_model'] = max(accuracies.items(), key=lambda x: x[1])[0]
            else:
                row['best_model'] = 'Finetune' if result['finetune_model']['metrics']['accuracy'] > trad_metrics['accuracy'] else 'Traditional'
        else:
            if has_combined:
                row['best_model'] = 'Finetune' if result['finetune_model']['metrics']['accuracy'] > result['combined_model']['metrics']['accuracy'] else 'Combined'
            else:
                row['best_model'] = 'Finetune'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
    
    with open(os.path.join(output_dir, 'full_comparison_results.json'), 'w') as f:
        json_ready_results = []
        for result in comparison_results:
            result_copy = result.copy()
            result_copy['network_limit'] = int(result_copy['network_limit'])
            result_copy['sample_count'] = int(result_copy['sample_count'])
            
            # Process finetune metrics
            finetune_metrics = result_copy['finetune_model']['metrics']
            finetune_metrics['accuracy'] = float(finetune_metrics['accuracy'])
            finetune_metrics['precision'] = float(finetune_metrics['precision'])
            finetune_metrics['recall'] = float(finetune_metrics['recall'])
            finetune_metrics['f1_score'] = float(finetune_metrics['f1_score'])
            finetune_metrics['roc_auc'] = float(finetune_metrics['roc_auc'])
            finetune_metrics['predictions'] = finetune_metrics['predictions']
            finetune_metrics['targets'] = finetune_metrics['targets']
            finetune_metrics['probabilities'] = finetune_metrics['probabilities']
            finetune_metrics['confusion_matrix'] = finetune_metrics['confusion_matrix'].tolist()
            
            # Process combined metrics if available
            if has_combined:
                combined_metrics = result_copy['combined_model']['metrics']
                combined_metrics['accuracy'] = float(combined_metrics['accuracy'])
                combined_metrics['precision'] = float(combined_metrics['precision'])
                combined_metrics['recall'] = float(combined_metrics['recall'])
                combined_metrics['f1_score'] = float(combined_metrics['f1_score'])
                combined_metrics['roc_auc'] = float(combined_metrics['roc_auc'])
                combined_metrics['predictions'] = combined_metrics['predictions']
                combined_metrics['targets'] = combined_metrics['targets']
                combined_metrics['probabilities'] = combined_metrics['probabilities']
                combined_metrics['confusion_matrix'] = combined_metrics['confusion_matrix'].tolist()
            
            # Process traditional metrics if available
            if has_traditional and result_copy.get('traditional_best', {}).get('exists', False):
                trad_metrics = result_copy['traditional_best']['metrics']
                trad_metrics['accuracy'] = float(trad_metrics['accuracy'])
                trad_metrics['f1_score'] = float(trad_metrics['f1_score'])
                trad_metrics['precision'] = float(trad_metrics['precision'])
                trad_metrics['recall'] = float(trad_metrics['recall'])
                trad_metrics['short_window'] = int(trad_metrics['short_window'])
                trad_metrics['long_window'] = int(trad_metrics['long_window'])
                trad_metrics['peak_tolerance'] = float(trad_metrics['peak_tolerance'])
                trad_metrics['count_threshold'] = int(trad_metrics['count_threshold'])
            
            json_ready_results.append(result_copy)
        
        json.dump(json_ready_results, f, indent=2)
    
    print("\nComparison complete! Results saved to:", output_dir)

    # Plotting - Generate comparison plots and ROC curves
    print("\nGenerating comparison plots...")
    print(f"Using custom heuristic parameters: {custom_heuristic_params}")
    
    # Determine which models are available for ROC curves
    model_types = ['finetune']
    if has_combined:
        model_types.append('combined')
    
    # Create ROC curve plots
    create_roc_curve_plot(comparison_results, output_dir, model_types)
    
    # Create comprehensive AUC visualizations
    print("\nGenerating AUC-specific visualizations...")
    try:
        create_auc_bar_chart(comparison_results, output_dir, has_combined)
    except Exception as e:
        print(f"Error creating AUC bar chart: {e}")
    
    try:
        create_auc_heatmap(comparison_results, output_dir, has_combined)
    except Exception as e:
        print(f"Error creating AUC heatmap: {e}")
    
    try:
        create_auc_distribution_plot(comparison_results, output_dir, has_combined)
    except Exception as e:
        print(f"Error creating AUC distribution plot: {e}")
    
    try:
        create_performance_radar_chart(comparison_results, output_dir, has_combined)
    except Exception as e:
        print(f"Error creating performance radar chart: {e}")
    
    try:
        create_auc_performance_table(comparison_results, output_dir, has_combined)
    except Exception as e:
        print(f"Error creating AUC performance table: {e}")
    
    # Create standard metric comparison plots
    print("\nGenerating standard metric comparison plots...")
    
    # Create simple line comparisons for accuracy and F1 score
    create_simple_line_comparison(comparison_results, output_dir, metric='accuracy', has_combined=has_combined)
    create_simple_line_comparison(comparison_results, output_dir, metric='f1_score', has_combined=has_combined)
    
    # Create detailed comparison plots with boxplots if traditional data available
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='accuracy', has_combined=has_combined, has_traditional=has_traditional,
                          custom_heuristic_params=custom_heuristic_params)
    
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='f1_score', has_combined=has_combined, has_traditional=has_traditional,
                          custom_heuristic_params=custom_heuristic_params)
    
    # Create AUC comparison plot
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='roc_auc', has_combined=has_combined, has_traditional=False)  # Traditional doesn't have AUC
    
    print("\nAll visualizations generated successfully!")
    print(f"\nGenerated visualization files:")
    print(f"  • ROC Curves: roc_curves_individual.png, roc_curves_aggregate.png")
    print(f"  • AUC Visualizations: auc_bar_chart.png, auc_heatmap.png, auc_distribution.png")
    print(f"  • Performance Analysis: performance_radar_chart.png, auc_performance_table.png")
    print(f"  • Simple Line Comparisons: accuracy_line_comparison.png, f1_score_line_comparison.png")
    print(f"  • Detailed Comparisons: accuracy_comparison_plot.png, f1_score_comparison_plot.png, roc_auc_comparison_plot.png")

if __name__ == "__main__":
    main()