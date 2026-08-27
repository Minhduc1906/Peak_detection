import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
import joblib
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

def preprocess_data_baseline(df, include_filename_features=False, include_direction=False, 
                           include_network_limit=True, scaler=None, train_mode=False):
    """
    Preprocess data for baseline models (no time sequences needed)
    """
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
    
    return X_scaled, y, feature_cols, scaler

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    time_steps = checkpoint.get('time_steps', 10)
    return state_dict, time_steps

def load_baseline_model(model_path):
    """Load a baseline model saved with joblib"""
    try:
        checkpoint = joblib.load(model_path)
        model = checkpoint['model']
        feature_names = checkpoint.get('feature_names', [])
        return model, feature_names
    except Exception as e:
        print(f"Error loading baseline model from {model_path}: {e}")
        return None, None

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

def evaluate_baseline_model(model, X, y):
    """Evaluate a baseline sklearn model"""
    y_pred = model.predict(X)
    
    # Get prediction probabilities (handle models that may not have predict_proba)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]  # Probability of positive class
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X)
            # Normalize decision function to [0,1] range
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            y_prob = y_pred.astype(float)  # Fallback to predictions
    except:
        y_prob = y_pred.astype(float)  # Fallback to predictions
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y, y_prob)
    except ValueError:
        # Handle case where all labels are the same class
        roc_auc = 0.5
    
    cm = confusion_matrix(y, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred.tolist(),
        'targets': y.tolist(),
        'probabilities': y_prob.tolist()
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

def create_roc_curve_plot(comparison_results, output_dir, model_types=['finetune', 'combined', 'random_forest', 'knn', 'gradient_boosting', 'decision_tree']):
    """
    Create ROC curve plots for different models
    """
    print(f"\nGenerating ROC curve plots...")
    
    # Define colors for each model type
    model_colors = {
        'finetune': 'green',
        'combined': 'purple', 
        'random_forest': 'blue',
        'knn': 'orange',
        'gradient_boosting': 'brown',
        'decision_tree': 'pink'
    }
    
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
            model_data = None
            color = model_colors.get(model_type, 'gray')
            
            if model_type == 'finetune':
                model_data = result.get('finetune_model')
                label = 'Finetune Model'
            elif model_type == 'combined' and 'combined_model' in result and result['combined_model'].get('exists', True):
                model_data = result.get('combined_model')
                label = 'Combined Model'
            elif model_type in result:  # Baseline models
                model_data = result.get(model_type)
                label = model_type.replace('_', ' ').title()
            else:
                continue
            
            if model_data and 'metrics' in model_data:
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
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(comparison_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_individual.png'), dpi=300, bbox_inches='tight')
    print(f"Individual ROC curves saved to {os.path.join(output_dir, 'roc_curves_individual.png')}")
    plt.close()
    
    # Create aggregate ROC curve plot
    plt.figure(figsize=(12, 10))
    
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
            
            color = model_colors.get(model_type, 'gray')
            label = model_type.replace('_', ' ').title()
            if model_type == 'finetune':
                label = 'Finetune Model'
            elif model_type == 'combined':
                label = 'Combined Model'
            
            plt.plot(mean_fpr, mean_tpr, color=color, linewidth=3, 
                    label=f'{label} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})')
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Mean ROC Curves with Confidence Intervals', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'roc_curves_aggregate.png'), dpi=300, bbox_inches='tight')
    print(f"Aggregate ROC curves saved to {os.path.join(output_dir, 'roc_curves_aggregate.png')}")
    plt.close()

def create_comparison_plot(comparison_results, param_df, output_dir, metric='accuracy', 
                          has_combined=True, has_traditional=True, custom_heuristic_params=None):
    """
    Create a comparison plot for a specific metric (accuracy, f1_score, or roc_auc) with traditional methods
    """
    print(f"\nGenerating {metric} comparison plot with traditional methods...")
    
    # Extract data for plotting
    testcase_labels = [r['scenario'] for r in comparison_results]
    finetune_values = [float(r['finetune_model']['metrics'][metric]) for r in comparison_results]
    
    # Add combined model values if available
    if has_combined:
        combined_values = [float(r['combined_model']['metrics'][metric]) for r in comparison_results]
    
    # Add baseline model values
    baseline_models = ['random_forest', 'knn', 'gradient_boosting', 'decision_tree']
    baseline_values = {}
    for model_name in baseline_models:
        values = []
        for result in comparison_results:
            if model_name in result and 'metrics' in result[model_name]:
                values.append(float(result[model_name]['metrics'][metric]))
            else:
                values.append(None)
        baseline_values[model_name] = values
    
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
        
        # Create plot without traditional data
        plt.figure(figsize=(18, 8))
        positions = range(len(testcase_labels))
        
        # Plot all models
        plt.plot(positions, finetune_values, 'g-o', label='Finetune Model', linewidth=2, markersize=8)
        
        if has_combined:
            plt.plot(positions, combined_values, 'purple', linestyle='--', marker='o', label='Combined Model', linewidth=2, markersize=8)
        
        # Plot baseline models
        colors = ['blue', 'orange', 'brown', 'pink']
        linestyles = ['-.', ':', '-', '-.']
        for i, (model_name, values) in enumerate(baseline_values.items()):
            valid_positions = [pos for pos, val in zip(positions, values) if val is not None]
            valid_values = [val for val in values if val is not None]
            if valid_values:
                plt.plot(valid_positions, valid_values, color=colors[i], linestyle=linestyles[i], 
                        marker='o', label=model_name.replace('_', ' ').title(), linewidth=2, markersize=6)
        
        plt.xticks(positions, testcase_labels, rotation=45, ha='right')
        plt.xlabel('Test Case')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison: All Models')
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

    # Create figure and axis
    plt.figure(figsize=(18, 8))
    positions = range(len(testcase_labels))

    # Box plots for traditional values per test case
    if total_data_points > 0:
        # Filter out empty lists for boxplot
        non_empty_positions = []
        non_empty_data = []

        for i, (pos, data) in enumerate(zip(positions, traditional_values_per_testcase)):
            if len(data) > 0:  # Only include if we have data
                non_empty_positions.append(pos)
                non_empty_data.append(data)

        if non_empty_data:
            print(f"Creating {metric} boxplots for {len(non_empty_data)} test cases with data")
            plt.boxplot(non_empty_data, positions=non_empty_positions, widths=0.6, patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', color='blue', alpha=0.7), 
                        whiskerprops=dict(color='blue'), capprops=dict(color='blue'), 
                        medianprops=dict(color='darkblue', linewidth=2))

    # Line plots for all models
    plt.plot(positions, finetune_values, 'g--o', label='Finetune Model', linewidth=2, markersize=8)

    if has_combined:
        plt.plot(positions, combined_values, 'purple', linestyle='-.', marker='o', label='Combined Model', linewidth=2, markersize=8)

    # Plot baseline models
    colors = ['blue', 'orange', 'brown', 'pink']
    linestyles = ['-.', ':', '-', '-.']
    for i, (model_name, values) in enumerate(baseline_values.items()):
        valid_positions = [pos for pos, val in zip(positions, values) if val is not None]
        valid_values = [val for val in values if val is not None]
        if valid_values:
            plt.plot(valid_positions, valid_values, color=colors[i], linestyle=linestyles[i], 
                    marker='o', label=model_name.replace('_', ' ').title(), linewidth=2, markersize=6)

    # Line plot for custom heuristic if available
    if has_custom_heuristic:
        plt.plot(positions, custom_heuristic_values, 'r-s', 
                label='Custom Heuristic', 
                linewidth=2, markersize=8)

    # Add legend entry for traditional method boxplots
    if total_data_points > 0:
        plt.plot([], [], 's', color='lightblue', label='Traditional (All Params)', markersize=10)

    # Customize plot
    plt.xticks(positions, testcase_labels, rotation=45, ha='right')
    plt.xlabel('Test Case')
    plt.ylabel(metric.replace('_', ' ').title())
    
    title_parts = [f'{metric.replace("_", " ").title()} Comparison:']
    if total_data_points > 0:
        title_parts.append('Traditional (Boxplots)')
    if has_custom_heuristic:
        title_parts.append('Custom Heuristic (Red)')
    title_parts.extend(['ML Models (Lines)'])
    
    plt.title(', '.join(title_parts))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plot_filename = f'{metric}_comparison_plot.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
    plt.close()

def create_model_performance_summary(comparison_results, output_dir, has_traditional=True):
    """
    Create a summary table and heatmap of all model performances including traditional methods
    """
    print(f"\nGenerating model performance summary...")
    
    # Extract performance data
    model_types = ['finetune_model', 'combined_model', 'random_forest', 'knn', 
                   'gradient_boosting', 'decision_tree']
    if has_traditional:
        model_types.append('traditional_best')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Calculate average performance across all test cases
    summary_data = []
    
    for model_type in model_types:
        if model_type == 'traditional_best':
            model_label = 'Traditional Heuristic'
        else:
            model_label = model_type.replace('_model', '').replace('_', ' ').title()
            if model_type == 'finetune_model':
                model_label = 'Finetune Model'
            elif model_type == 'combined_model':
                model_label = 'Combined Model'
        
        model_metrics = {}
        for metric in metrics:
            values = []
            for result in comparison_results:
                if model_type == 'traditional_best':
                    # Handle traditional methods (only have accuracy, f1_score, precision, recall)
                    if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                        if result.get('traditional_best', {}).get('exists', False):
                            trad_result = result['traditional_best']['metrics']
                            if metric in trad_result:
                                values.append(float(trad_result[metric]))
                    # Traditional methods don't have roc_auc, skip
                    elif metric == 'roc_auc':
                        continue
                elif model_type in result and 'metrics' in result[model_type]:
                    values.append(float(result[model_type]['metrics'][metric]))
                elif model_type == 'combined_model' and not result.get('combined_model', {}).get('exists', True):
                    continue  # Skip if combined model doesn't exist
            
            if values:
                model_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
            else:
                model_metrics[metric] = {'mean': 0, 'std': 0, 'count': 0}
        
        # Only add model if it has any valid metrics
        if any(model_metrics[m]['count'] > 0 for m in model_metrics):
            summary_entry = {
                'model': model_label,
                'model_type': model_type,
                'test_cases': max(model_metrics[m]['count'] for m in model_metrics if model_metrics[m]['count'] > 0)
            }
            
            # Add mean and std for each metric
            for metric in metrics:
                if metric in model_metrics:
                    summary_entry[f'{metric}_mean'] = model_metrics[metric]['mean']
                    summary_entry[f'{metric}_std'] = model_metrics[metric]['std']
                else:
                    summary_entry[f'{metric}_mean'] = 0
                    summary_entry[f'{metric}_std'] = 0
            
            summary_data.append(summary_entry)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary CSV
    summary_df.to_csv(os.path.join(output_dir, 'model_performance_summary.csv'), index=False)
    
    # Create heatmap of mean performance - FIXED VERSION
    heatmap_data = []
    model_labels = []
    
    # First pass: determine if we have any traditional methods
    has_traditional_data = any(row['model_type'] == 'traditional_best' and row['test_cases'] > 0 for row in summary_data)
    
    # Set metrics based on what we have
    if has_traditional_data:
        # If we have traditional methods, use all metrics but pad traditional with 0 for ROC_AUC
        heatmap_metrics = metrics  # ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    else:
        # If no traditional methods, use all metrics
        heatmap_metrics = metrics
    
    for row in summary_data:
        if row['test_cases'] > 0:  # Only include models with data
            if row['model_type'] == 'traditional_best':
                # Traditional only has accuracy, f1_score, precision, recall - pad with 0 for ROC_AUC
                metric_values = [
                    row['accuracy_mean'],
                    row['precision_mean'], 
                    row['recall_mean'],
                    row['f1_score_mean'],
                    0.0  # ROC_AUC not available for traditional
                ]
            else:
                # ML models have all metrics
                metric_values = [row[f'{metric}_mean'] for metric in heatmap_metrics]
            
            heatmap_data.append(metric_values)
            model_labels.append(row['model'])
    
    if heatmap_data:
        # Now all rows should have the same length
        heatmap_array = np.array(heatmap_data)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(model_labels) * 0.5)))
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(heatmap_metrics)))
        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in heatmap_metrics])
        ax.set_yticklabels(model_labels)
        
        # Add text annotations
        for i in range(len(model_labels)):
            for j in range(len(heatmap_metrics)):
                value = heatmap_array[i, j]
                # Show "N/A" for traditional ROC_AUC (which is 0)
                if j == 4 and model_labels[i] == 'Traditional Heuristic' and value == 0:
                    text_val = "N/A"
                    color = "gray"
                else:
                    text_val = f'{value:.3f}'
                    color = "black"
                
                text = ax.text(j, i, text_val, ha="center", va="center", 
                             color=color, fontweight='bold')
        
        ax.set_title("Model Performance Heatmap (Mean Values)", fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Models', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to {os.path.join(output_dir, 'model_performance_heatmap.png')}")
        plt.close()
    
    return summary_df

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
    
    # Load baseline models
    baseline_models = {}
    baseline_model_names = ['random_forest', 'knn', 'gradient_boosting', 'decision_tree']
    
    for model_name in baseline_model_names:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            model, feature_names = load_baseline_model(model_path)
            if model is not None:
                baseline_models[model_name] = {'model': model, 'feature_names': feature_names}
                print(f"Loaded baseline model: {model_name}")
            else:
                print(f"Failed to load baseline model: {model_name}")
        else:
            print(f"Baseline model not found: {model_path}")
    
    print(f"Loaded {len(baseline_models)} baseline models: {list(baseline_models.keys())}")
    
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
        X_test_seq, y_test_seq, feature_cols, _ = preprocess_data(
            test_data, include_network_limit=True, time_steps=finetune_time_steps
        )
        finetune_model = LSTMModel(input_size=X_test_seq.shape[2])
        finetune_model.load_state_dict(finetune_state_dict)
        finetune_model.to(device)
        finetune_metrics = evaluate_model(finetune_model, X_test_seq, y_test_seq)
        results_entry['finetune_model'] = {'metrics': finetune_metrics}
        print(f"  Finetune model accuracy: {finetune_metrics['accuracy']:.2f}, AUC: {finetune_metrics['roc_auc']:.2f}")
        
        # Evaluate combined model if available
        if has_combined:
            print(f"  Evaluating combined model")
            X_test_seq, y_test_seq, feature_cols, _ = preprocess_data(
                test_data, include_network_limit=True, time_steps=combined_time_steps
            )
            combined_model = LSTMModel(input_size=X_test_seq.shape[2])
            combined_model.load_state_dict(combined_state_dict)
            combined_model.to(device)
            combined_metrics = evaluate_model(combined_model, X_test_seq, y_test_seq)
            results_entry['combined_model'] = {'metrics': combined_metrics}
            print(f"  Combined model accuracy: {combined_metrics['accuracy']:.2f}, AUC: {combined_metrics['roc_auc']:.2f}")
        else:
            results_entry['combined_model'] = {'exists': False}
        
        # Evaluate baseline models
        if baseline_models:
            print(f"  Evaluating baseline models")
            # Preprocess data for baseline models (no sequences)
            X_test_baseline, y_test_baseline, _, _ = preprocess_data_baseline(
                test_data, include_network_limit=True
            )
            
            for model_name, model_info in baseline_models.items():
                try:
                    model = model_info['model']
                    baseline_metrics = evaluate_baseline_model(model, X_test_baseline, y_test_baseline)
                    results_entry[model_name] = {'metrics': baseline_metrics}
                    print(f"  {model_name} accuracy: {baseline_metrics['accuracy']:.2f}, AUC: {baseline_metrics['roc_auc']:.2f}")
                except Exception as e:
                    print(f"  Error evaluating {model_name}: {e}")
                    results_entry[model_name] = {'error': str(e)}
        
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
    # Create comprehensive summary including all models
    summary_data = []
    for result in comparison_results:
        row = {
            'testcase': result['testcase'],
            'scenario': result['scenario'],
            'network_limit': result['network_limit'],
            'sample_count': result['sample_count'],
        }
        
        # Add finetune model metrics
        if 'finetune_model' in result:
            finetune_metrics = result['finetune_model']['metrics']
            row.update({
                'finetune_accuracy': float(finetune_metrics['accuracy']),
                'finetune_f1': float(finetune_metrics['f1_score']),
                'finetune_auc': float(finetune_metrics['roc_auc']),
            })
        
        # Add combined model metrics if available
        if has_combined and result.get('combined_model', {}).get('exists', True):
            comb_metrics = result['combined_model']['metrics']
            row.update({
                'combined_accuracy': float(comb_metrics['accuracy']),
                'combined_f1': float(comb_metrics['f1_score']),
                'combined_auc': float(comb_metrics['roc_auc']),
            })
        
        # Add baseline model metrics
        for model_name in baseline_model_names:
            if model_name in result and 'metrics' in result[model_name]:
                metrics = result[model_name]['metrics']
                row.update({
                    f'{model_name}_accuracy': float(metrics['accuracy']),
                    f'{model_name}_f1': float(metrics['f1_score']),
                    f'{model_name}_auc': float(metrics['roc_auc']),
                })
        
        # Add traditional metrics if available
        if has_traditional and result.get('traditional_best', {}).get('exists', False):
            trad_metrics = result['traditional_best']['metrics']
            row.update({
                'traditional_accuracy': float(trad_metrics['accuracy']),
                'traditional_f1': float(trad_metrics['f1_score']),
            })
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'comprehensive_model_comparison.csv'), index=False)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'full_comparison_results_with_baselines.json'), 'w') as f:
        json_ready_results = []
        for result in comparison_results:
            result_copy = result.copy()
            result_copy['network_limit'] = int(result_copy['network_limit'])
            result_copy['sample_count'] = int(result_copy['sample_count'])
            
            # Process all model metrics for JSON serialization
            for model_key in result_copy.keys():
                if isinstance(result_copy[model_key], dict) and 'metrics' in result_copy[model_key]:
                    metrics = result_copy[model_key]['metrics']
                    # Convert numpy arrays and ensure all values are JSON serializable
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            metrics[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            metrics[key] = float(value)
            
            # Process traditional metrics if available
            if has_traditional and result_copy.get('traditional_best', {}).get('exists', False):
                trad_metrics = result_copy['traditional_best']['metrics']
                for key, value in trad_metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        trad_metrics[key] = float(value)
            
            json_ready_results.append(result_copy)
        
        json.dump(json_ready_results, f, indent=2)
    
    print("\nComparison complete! Results saved to:", output_dir)

    # Generate comprehensive visualizations
    print("\nGenerating comprehensive comparison plots...")
    
    # Determine which models are available
    available_models = ['finetune']
    if has_combined:
        available_models.append('combined')
    
    # Add available baseline models
    for model_name in baseline_model_names:
        if any(model_name in result for result in comparison_results):
            available_models.append(model_name)
    
    print(f"Available models for plotting: {available_models}")
    
    # Create ROC curve plots with all models
    create_roc_curve_plot(comparison_results, output_dir, available_models)
    
    # Create detailed comparison plots with traditional method boxplots
    print("\nGenerating detailed comparison plots with traditional method boxplots...")
    
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
    
    # Create model performance summary and heatmap
    print("\nGenerating model performance summary and heatmap...")
    summary_df = create_model_performance_summary(comparison_results, output_dir, has_traditional)
    
    print("\nAll visualizations generated successfully!")
    print(f"\nGenerated visualization files:")
    print(f"  • ROC Curves: roc_curves_individual.png, roc_curves_aggregate.png")
    print(f"  • Detailed Comparisons with Traditional Boxplots: accuracy_comparison_plot.png, f1_score_comparison_plot.png, roc_auc_comparison_plot.png")
    print(f"  • Performance Summary with Heuristic: model_performance_heatmap.png, model_performance_summary.csv")
    print(f"  • Detailed Results: comprehensive_model_comparison.csv, full_comparison_results_with_baselines.json")
    print(f"\nColor Scheme:")
    print(f"  🟢 Green = Finetune Model")
    print(f"  🟣 Purple = Combined Model") 
    print(f"  🔴 Red = Custom Heuristic")
    print(f"  🔵 Blue = Random Forest")
    print(f"  🟠 Orange = K-NN")
    print(f"  🟤 Brown = Gradient Boosting")
    print(f"  🩷 Pink = Decision Tree")
    print(f"  📦 Light Blue Boxes = Traditional Heuristic (All Parameters)")
    
    # Print final summary
    if not summary_df.empty:
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Mean Acc':<10} {'Mean AUC':<10} {'Mean F1':<10} {'Test Cases':<12}")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            if row['test_cases'] > 0:
                print(f"{row['model']:<20} {row['accuracy_mean']:<10.3f} {row['roc_auc_mean']:<10.3f} {row['f1_score_mean']:<10.3f} {row['test_cases']:<12}")

if __name__ == "__main__":
    main()