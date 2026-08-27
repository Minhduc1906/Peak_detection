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
from scipy import stats

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.8,
    'figure.autolayout': True,
})

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

def extract_network_limit(subfolder):
    match = re.search(r'(\d+)$', subfolder)
    return int(match.group(1)) if match else -1

def extract_scenario_name(filename):
    base_name = filename.replace('.csv', '').replace('test_', '')
    base_name = re.sub(r'_s2.*$', '', base_name)
    base_name = re.sub(r'_(downstream|upstream)$', '', base_name)
    return base_name

def calculate_rate_matlab_method(df, byte_col, time_col='relative_time'):
    if byte_col not in df.columns or 'tx_bytes' not in df.columns:
        return np.zeros(len(df))
    
    time_values = df[time_col].values
    if len(time_values) < 3:
        return np.zeros(len(df))
    
    relative_time = time_values[1:] - time_values[1]
    if len(relative_time) < 2:
        return np.zeros(len(df))
    
    interval = np.mean(np.diff(relative_time))
    if interval <= 0:
        interval = 0.02
    
    byte_values = df['tx_bytes'].values[1:]
    bit_rate = 8 * byte_values / interval / 1e6
    bit_rate_full = np.concatenate([np.zeros(1), bit_rate])
    bit_rate_full = np.maximum(bit_rate_full, 0)
    
    return bit_rate_full

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

def preprocess_data(df, include_filename_features=False, include_direction=False, 
                   include_network_limit=True, time_steps=10, scaler=None, train_mode=False):
    data_copy = df.copy()
    
    for col in data_copy.columns:
        if data_copy[col].dtype.kind in 'if':
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
    data_copy = df.copy()
    
    for col in data_copy.columns:
        if data_copy[col].dtype.kind in 'if':
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
    try:
        checkpoint = joblib.load(model_path)
        model = checkpoint['model']
        feature_names = checkpoint.get('feature_names', [])
        return model, feature_names
    except Exception as e:
        print(f"Error loading baseline model from {model_path}: {e}")
        return None, None

def find_raw_data_file(filename, network_limit):
    """Helper function to find raw data file"""
    extracted_data_folder = 'extracted_data'
    
    for root, dirs, files in os.walk(extracted_data_folder):
        for file in files:
            if file.endswith('_downstream.csv') and filename in file:
                subfolder = os.path.basename(root)
                if extract_network_limit(subfolder) == network_limit:
                    return os.path.join(root, file)
    return None

def evaluate_model(model, X, y, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
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
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            
            all_preds.extend(predicted.cpu().flatten().tolist())
            all_targets.extend(y_batch.cpu().flatten().tolist())
            all_probs.extend(outputs.cpu().flatten().tolist())
    
    all_preds_tensor = torch.tensor(all_preds)
    all_targets_tensor = torch.tensor(all_targets)
    
    correct = (all_preds_tensor == all_targets_tensor).sum().item()
    total = len(all_targets_tensor)
    acc = correct / total if total > 0 else 0
    
    all_preds_list = all_preds
    all_targets_list = all_targets
    all_probs_list = all_probs
    
    prec = precision_score(all_targets_list, all_preds_list, zero_division=0)
    rec = recall_score(all_targets_list, all_preds_list, zero_division=0)
    f1 = f1_score(all_targets_list, all_preds_list, zero_division=0)
    cm = confusion_matrix(all_targets_list, all_preds_list)
    
    try:
        roc_auc = roc_auc_score(all_targets_list, all_probs_list)
    except ValueError:
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
        'probabilities': all_probs_list
    }

def evaluate_baseline_model(model, X, y):
    y_pred = model.predict(X)
    
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            y_prob = y_pred.astype(float)
    except:
        y_prob = y_pred.astype(float)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y, y_prob)
    except ValueError:
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

def analyze_congestion_throughput_improved(comparison_results, feature_data, output_dir):
    """
    Enhanced analysis that directly compares throughput during predicted vs ground truth congestion.
    Focuses on evaluating if models detect congestion when throughput is similar to ground truth congestion.
    NBN policy: actual network limit = stated limit + 15%
    """
    print("\nAnalyzing throughput during predicted vs ground truth congestion periods...")
    print("Focus: Do models detect congestion when throughput matches ground truth congestion patterns?")
    print("Note: Using NBN policy - actual network limit = stated limit + 15%")
    
    congestion_analysis = []
    
    for result in comparison_results:
        testcase = result['testcase']
        scenario = result['scenario']
        network_limit = result['network_limit']
        actual_network_limit = network_limit * 1.15  # NBN policy
        
        print(f"\nAnalyzing {testcase}...")
        print(f"  Stated limit: {network_limit} Mbps, Actual NBN limit: {actual_network_limit:.1f} Mbps")
        
        # Parse testcase to get filename and network limit
        filename, _ = testcase.rsplit('_', 1)
        
        # Load raw data from extracted_data folder to get actual throughput
        raw_data_path = find_raw_data_file(filename, network_limit)
        
        if not raw_data_path or not os.path.exists(raw_data_path):
            print(f"  Warning: Raw data file not found for {testcase}")
            continue
        
        print(f"  Loading raw data from: {raw_data_path}")
        
        try:
            raw_df = pd.read_csv(raw_data_path)
            
            # Check for required columns
            if 'tx_bytes' not in raw_df.columns:
                print(f"  Error: tx_bytes column not found in raw data")
                continue
                
            # Calculate time column
            time_cols = [col for col in raw_df.columns if 'time' in col.lower()]
            if 'relative_time' in raw_df.columns:
                time_col = 'relative_time'
            elif 'time' in raw_df.columns:
                time_col = 'time'
                raw_df['relative_time'] = raw_df['time'] - raw_df['time'].iloc[0]
                time_col = 'relative_time'
            elif time_cols:
                time_col = time_cols[0]
                raw_df['relative_time'] = raw_df[time_col] - raw_df[time_col].iloc[0]
                time_col = 'relative_time'
            else:
                print(f"  Error: No time column found")
                continue
            
            # Calculate downstream throughput using MATLAB method
            throughput = calculate_rate_matlab_method(raw_df, 'tx_bytes', time_col)
            
            if throughput.max() == 0:
                print(f"  Warning: All throughput values are zero - check raw data")
                continue
                
        except Exception as e:
            print(f"  Error loading raw data: {e}")
            continue
        
        # Get feature data for this testcase to align predictions
        feature_test_data = feature_data[
            (feature_data['file_name'] == filename) & 
            (feature_data['network_limit'] == network_limit) &
            (feature_data['direction'] == 'downstream')
        ]
        
        if len(feature_test_data) == 0:
            print(f"  Warning: No feature data found for {testcase}")
            continue
        
        # Align data lengths - use minimum length
        min_length = min(len(throughput), len(feature_test_data))
        throughput_aligned = throughput[:min_length]
        
        # Get ground truth congestion periods and their throughput
        ground_truth_congested = feature_test_data['queue_exists'].values[:min_length] == 1
        ground_truth_throughput = throughput_aligned[ground_truth_congested]
        
        if len(ground_truth_throughput) == 0:
            print(f"  No ground truth congestion periods found")
            continue
            
        print(f"  Ground truth congestion: {len(ground_truth_throughput)} samples")
        print(f"  Ground truth throughput stats: min={ground_truth_throughput.min():.2f}, "
              f"max={ground_truth_throughput.max():.2f}, mean={ground_truth_throughput.mean():.2f}")
        
        analysis_entry = {
            'testcase': testcase,
            'scenario': scenario,
            'network_limit': network_limit,
            'actual_network_limit': actual_network_limit,
            'total_samples': min_length,
            'avg_throughput': throughput_aligned.mean(),
            'max_throughput': throughput_aligned.max(),
            'ground_truth_congested_samples': len(ground_truth_throughput),
            'ground_truth_mean_throughput': float(ground_truth_throughput.mean()),
            'ground_truth_std_throughput': float(ground_truth_throughput.std()),
            'ground_truth_capacity_percentage': float((ground_truth_throughput.mean() / actual_network_limit) * 100),
        }
        
        models_to_analyze = ['finetune_model', 'combined_model', 'random_forest', 'knn', 'decision_tree']
        
        for model_name in models_to_analyze:
            if model_name not in result or 'metrics' not in result[model_name]:
                continue
                
            predictions = np.array(result[model_name]['metrics']['predictions'])
            
            # Handle LSTM sequence offset more carefully
            if model_name in ['finetune_model', 'combined_model']:
                time_steps = 10
                # Predictions are shorter due to sequence creation
                if len(predictions) < min_length:
                    # Create aligned predictions array
                    aligned_predictions = np.zeros(min_length)
                    if len(predictions) > 0:
                        # Place predictions at the end (after time_steps offset)
                        start_idx = min_length - len(predictions)
                        aligned_predictions[start_idx:] = predictions
                else:
                    aligned_predictions = predictions[:min_length]
            else:
                aligned_predictions = predictions[:min_length]
            
            # Get predicted congestion periods and their throughput
            predicted_congested = aligned_predictions == 1
            predicted_throughput = throughput_aligned[predicted_congested]
            
            if len(predicted_throughput) == 0:
                print(f"  {model_name}: No congestion predicted")
                analysis_entry.update({
                    f'{model_name}_congested_samples': 0,
                    f'{model_name}_mean_throughput': 0.0,
                    f'{model_name}_capacity_percentage': 0.0,
                    f'{model_name}_throughput_similarity_to_gt': 0.0,
                    f'{model_name}_distribution_similarity_to_gt': 0.0,
                    f'{model_name}_overlap_with_gt_range_pct': 0.0,
                    f'{model_name}_mean_diff_from_gt': float(ground_truth_throughput.mean()),
                })
                continue
            
            print(f"  {model_name}: {len(predicted_throughput)} congested samples")
            print(f"  {model_name} throughput stats: min={predicted_throughput.min():.2f}, "
                  f"max={predicted_throughput.max():.2f}, mean={predicted_throughput.mean():.2f}")
            
            # Calculate similarity metrics between predicted and ground truth throughput
            # 1. Mean difference (how close are the average throughputs?)
            mean_diff = abs(predicted_throughput.mean() - ground_truth_throughput.mean())
            mean_similarity = 1.0 / (1.0 + mean_diff / ground_truth_throughput.mean())
            
            # 2. Distribution similarity using KS test
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(predicted_throughput, ground_truth_throughput)
                distribution_similarity = 1.0 - ks_stat  # Higher is more similar
            except:
                distribution_similarity = 0.0
            
            # 3. Overlap analysis - what percentage of predicted congestion has throughput 
            # within the range of ground truth congestion?
            gt_min, gt_max = ground_truth_throughput.min(), ground_truth_throughput.max()
            within_gt_range = np.sum((predicted_throughput >= gt_min) & (predicted_throughput <= gt_max))
            overlap_percentage = within_gt_range / len(predicted_throughput) * 100
            
            # Capacity percentage using actual NBN network limit
            capacity_percentage = (predicted_throughput.mean() / actual_network_limit) * 100
            
            analysis_entry.update({
                f'{model_name}_congested_samples': int(len(predicted_throughput)),
                f'{model_name}_mean_throughput': float(predicted_throughput.mean()),
                f'{model_name}_std_throughput': float(predicted_throughput.std()),
                f'{model_name}_capacity_percentage': float(capacity_percentage),
                f'{model_name}_throughput_similarity_to_gt': float(mean_similarity),
                f'{model_name}_distribution_similarity_to_gt': float(distribution_similarity),
                f'{model_name}_overlap_with_gt_range_pct': float(overlap_percentage),
                f'{model_name}_mean_diff_from_gt': float(mean_diff),
            })
            
            print(f"  {model_name} vs Ground Truth Comparison:")
            print(f"    Mean throughput similarity: {mean_similarity:.3f}")
            print(f"    Distribution similarity: {distribution_similarity:.3f}")
            print(f"    Overlap with GT range: {overlap_percentage:.1f}%")
            print(f"    Mean difference: {mean_diff:.2f} Mbps")
            print(f"    Capacity percentage (NBN adjusted): {capacity_percentage:.1f}%")
        
        congestion_analysis.append(analysis_entry)
    
    if congestion_analysis:
        congestion_df = pd.DataFrame(congestion_analysis)
        
        # Save both original and improved analysis
        congestion_csv_path = os.path.join(output_dir, 'congestion_throughput_analysis.csv')
        congestion_df.to_csv(congestion_csv_path, index=False)
        
        # Save detailed similarity analysis
        similarity_csv_path = os.path.join(output_dir, 'congestion_throughput_similarity_analysis.csv')
        congestion_df.to_csv(similarity_csv_path, index=False)
        
        print(f"\nCongestion analysis saved to: {congestion_csv_path}")
        print(f"Detailed similarity analysis saved to: {similarity_csv_path}")
        
        return congestion_df
    
    return None

def get_traditional_best_accuracy(param_tests_file, testcase):
    try:
        param_df = pd.read_csv(param_tests_file)
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        expected_downstream = f"{filename}_downstream.csv"
        
        case_params = param_df[
            (param_df['file_name'] == expected_downstream) & 
            (param_df['network_limit'] == network_limit) &
            (param_df['direction'] == 'downstream')
        ]
        
        if case_params.empty:
            case_params = param_df[
                (param_df['file_name'].str.contains(filename, na=False, case=False)) & 
                (param_df['network_limit'] == network_limit) &
                (param_df['direction'] == 'downstream')
            ]
        
        if case_params.empty:
            return None
        
        successful_tests = case_params[case_params['status'] == 'success']
        if successful_tests.empty:
            return None
        
        best_row = successful_tests.loc[successful_tests['accuracy'].idxmax()]
        
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
        print(f"Error getting traditional best accuracy: {e}")
        return None

def get_custom_heuristic_accuracy(param_tests_file, testcase, custom_params):
    try:
        param_df = pd.read_csv(param_tests_file)
        
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        expected_downstream = f"{filename}_downstream.csv"
        
        case_params = param_df[
            (param_df['file_name'] == expected_downstream) & 
            (param_df['network_limit'] == network_limit) &
            (param_df['direction'] == 'downstream') &
            (param_df['status'] == 'success')
        ]
        
        if case_params.empty:
            case_params = param_df[
                (param_df['file_name'].str.contains(filename, na=False, case=False)) & 
                (param_df['network_limit'] == network_limit) &
                (param_df['direction'] == 'downstream') &
                (param_df['status'] == 'success')
            ]
        
        if case_params.empty:
            return None
        
        exact_match = case_params[
            (case_params['short_window'] == custom_params['short_window']) &
            (case_params['long_window'] == custom_params['long_window']) &
            (case_params['peak_tolerance'] == custom_params['peak_tolerance']) &
            (case_params['count_threshold'] == custom_params['count_threshold'])
        ]
        
        if not exact_match.empty:
            result_row = exact_match.iloc[0]
        else:
            case_params = case_params.copy()
            case_params['param_distance'] = (
                abs(case_params['short_window'] - custom_params['short_window']) +
                abs(case_params['long_window'] - custom_params['long_window']) +
                abs(case_params['peak_tolerance'] - custom_params['peak_tolerance']) * 100 +
                abs(case_params['count_threshold'] - custom_params['count_threshold'])
            )
            
            closest_match = case_params.loc[case_params['param_distance'].idxmin()]
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
        print(f"Error getting custom heuristic accuracy: {e}")
        return None
    
def create_congestion_throughput_plots(congestion_df, output_dir):
    """Create comprehensive plots including original and new similarity analysis"""
    plot_df = congestion_df[congestion_df['total_samples'] > 0].copy()
    
    if len(plot_df) == 0:
        print("No valid data for congestion throughput plots")
        return
    
    models = ['finetune_model', 'combined_model', 'random_forest', 'knn', 'decision_tree']
    model_labels = ['Finetune Model', 'Combined Model', 'Random Forest', 'K-NN', 'Decision Tree']
    colors = ['green', 'purple', 'blue', 'orange', 'pink']
    
    scenarios = plot_df['scenario'].tolist()
    x_pos = np.arange(len(scenarios))
    bar_width = 0.12
    
    # Plot 1: Original throughput comparison with NBN capacity
    plt.figure(figsize=(15, 8))
    actual_network_limits = plot_df['actual_network_limit'].tolist()
    
    plt.bar(x_pos, actual_network_limits, alpha=0.3, color='gray', label='Actual NBN Network Capacity', width=0.8)
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        throughput_col = f'{model}_mean_throughput'
        if throughput_col in plot_df.columns:
            values = plot_df[throughput_col].fillna(0).tolist()
            offset = (i - len(models)/2) * bar_width
            plt.bar(x_pos + offset, values, bar_width, label=label, color=color, alpha=0.7)
    
    if 'ground_truth_mean_throughput' in plot_df.columns:
        true_values = plot_df['ground_truth_mean_throughput'].fillna(0).tolist()
        offset = len(models) * bar_width / 2
        plt.bar(x_pos + offset, true_values, bar_width, label='Ground Truth', color='red', alpha=0.7)
    
    plt.xlabel('Test Scenarios')
    plt.ylabel('Average Throughput (Mbps)')
    plt.title('Average Throughput During Predicted Congestion Periods\n(Gray bars show actual NBN capacity = stated + 15%)')
    plt.xticks(x_pos, scenarios, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'congestion_throughput_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Congestion throughput comparison plot saved")
    plt.close()
    
    # Plot 2: Capacity utilization
    plt.figure(figsize=(15, 8))
    
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% of Actual NBN Capacity')
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        capacity_col = f'{model}_capacity_percentage'
        if capacity_col in plot_df.columns:
            values = plot_df[capacity_col].fillna(0).tolist()
            offset = (i - len(models)/2) * bar_width
            plt.bar(x_pos + offset, values, bar_width, label=label, color=color, alpha=0.7)
    
    if 'ground_truth_capacity_percentage' in plot_df.columns:
        true_capacity_values = plot_df['ground_truth_capacity_percentage'].fillna(0).tolist()
        offset = len(models) * bar_width / 2
        plt.bar(x_pos + offset, true_capacity_values, bar_width, label='Ground Truth', color='red', alpha=0.7)
    
    plt.xlabel('Test Scenarios')
    plt.ylabel('NBN Capacity Utilization (%)')
    plt.title('NBN Capacity Utilization During Predicted Congestion\n(Based on actual NBN limits = stated + 15%)')
    plt.xticks(x_pos, scenarios, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'congestion_capacity_utilization.png'), dpi=300, bbox_inches='tight')
    print(f"Congestion capacity utilization plot saved")
    plt.close()

    # Plot 3: NEW - Throughput Similarity Scores
    plt.figure(figsize=(15, 8))
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        similarity_col = f'{model}_throughput_similarity_to_gt'
        if similarity_col in plot_df.columns:
            values = plot_df[similarity_col].fillna(0).tolist()
            offset = (i - len(models)/2) * bar_width
            plt.bar(x_pos + offset, values, bar_width, label=label, color=color, alpha=0.7)
    
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good Similarity (0.8)')
    plt.xlabel('Test Scenarios')
    plt.ylabel('Throughput Similarity Score')
    plt.title('Model Throughput Similarity to Ground Truth Congestion\n(Higher is better - closer to GT congestion throughput)')
    plt.xticks(x_pos, scenarios, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'throughput_similarity_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Throughput similarity comparison plot saved")
    plt.close()
    
    # Plot 4: NEW - Overlap with Ground Truth Range
    plt.figure(figsize=(15, 8))
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        overlap_col = f'{model}_overlap_with_gt_range_pct'
        if overlap_col in plot_df.columns:
            values = plot_df[overlap_col].fillna(0).tolist()
            offset = (i - len(models)/2) * bar_width
            plt.bar(x_pos + offset, values, bar_width, label=label, color=color, alpha=0.7)
    
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Good Overlap (70%)')
    plt.xlabel('Test Scenarios')
    plt.ylabel('Overlap with GT Range (%)')
    plt.title('Percentage of Predicted Congestion Within Ground Truth Throughput Range\n(Higher is better - predicted congestion has similar throughput to GT)')
    plt.xticks(x_pos, scenarios, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'throughput_overlap_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Throughput overlap comparison plot saved")
    plt.close()
    
    # Plot 5: NEW - Distribution Similarity
    plt.figure(figsize=(15, 8))
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        dist_sim_col = f'{model}_distribution_similarity_to_gt'
        if dist_sim_col in plot_df.columns:
            values = plot_df[dist_sim_col].fillna(0).tolist()
            offset = (i - len(models)/2) * bar_width
            plt.bar(x_pos + offset, values, bar_width, label=label, color=color, alpha=0.7)
    
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good Distribution Similarity (0.7)')
    plt.xlabel('Test Scenarios')
    plt.ylabel('Distribution Similarity Score')
    plt.title('Throughput Distribution Similarity to Ground Truth Congestion\n(Higher is better - predicted congestion distribution matches GT)')
    plt.xticks(x_pos, scenarios, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'distribution_similarity_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Distribution similarity comparison plot saved")
    plt.close()

def create_roc_curve_plot(comparison_results, output_dir, model_types=['finetune', 'combined', 'random_forest', 'knn', 'decision_tree']):
    print(f"\nGenerating ROC curve plots...")
    
    model_colors = {
        'finetune': 'green',
        'combined': 'purple', 
        'random_forest': 'blue',
        'knn': 'orange',
        'decision_tree': 'pink'
    }
    
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
        
        for model_type in model_types:
            model_data = None
            color = model_colors.get(model_type, 'gray')
            
            if model_type == 'finetune':
                model_data = result.get('finetune_model')
                label = 'Finetune Model'
            elif model_type == 'combined' and 'combined_model' in result and result['combined_model'].get('exists', True):
                model_data = result.get('combined_model')
                label = 'Combined Model'
            elif model_type in result:
                model_data = result.get(model_type)
                label = model_type.replace('_', ' ').title()
            else:
                continue
            
            if model_data and 'metrics' in model_data:
                targets = model_data['metrics']['targets']
                probabilities = model_data['metrics']['probabilities']
                roc_auc = model_data['metrics']['roc_auc']
                
                fpr, tpr, _ = roc_curve(targets, probabilities)
                
                all_fpr[model_type].append(fpr)
                all_tpr[model_type].append(tpr)
                all_aucs[model_type].append(roc_auc)
                
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{label} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {scenario}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(comparison_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_individual.png'), dpi=300, bbox_inches='tight')
    print(f"Individual ROC curves saved")
    plt.close()
    
    # Aggregate ROC curve
    plt.figure(figsize=(12, 10))
    
    for model_type in model_types:
        if not all_aucs[model_type]:
            continue
            
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curves with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'roc_curves_aggregate.png'), dpi=300, bbox_inches='tight')
    print(f"Aggregate ROC curves saved")
    plt.close()

def create_comparison_plot(comparison_results, param_df, output_dir, metric='accuracy', 
                          has_combined=True, has_traditional=True, custom_heuristic_params=None):
    print(f"\nGenerating {metric} comparison plot...")
    
    testcase_labels = [r['scenario'] for r in comparison_results]
    finetune_values = [float(r['finetune_model']['metrics'][metric]) for r in comparison_results]
    
    if has_combined:
        combined_values = [float(r['combined_model']['metrics'][metric]) for r in comparison_results]
    
    baseline_models = ['random_forest', 'knn', 'decision_tree']
    baseline_values = {}
    for model_name in baseline_models:
        values = []
        for result in comparison_results:
            if model_name in result and 'metrics' in result[model_name]:
                values.append(float(result[model_name]['metrics'][metric]))
            else:
                values.append(None)
        baseline_values[model_name] = values
    
    custom_heuristic_values = []
    has_custom_heuristic = False
    if custom_heuristic_params and has_traditional and metric in ['accuracy', 'f1_score']:
        has_custom_heuristic = True
        
        for result in comparison_results:
            testcase = result['testcase']
            param_tests_file = 'combined_param_tests.csv'
            custom_result = get_custom_heuristic_accuracy(param_tests_file, testcase, custom_heuristic_params)
            
            if custom_result:
                custom_heuristic_values.append(float(custom_result[metric]))
            else:
                custom_heuristic_values.append(0.0)
    
    plt.figure(figsize=(18, 8))
    positions = range(len(testcase_labels))
    
    plt.plot(positions, finetune_values, 'g-o', label='Finetune Model', linewidth=2, markersize=8)
    
    if has_combined:
        plt.plot(positions, combined_values, 'purple', linestyle='--', marker='o', label='Combined Model', linewidth=2, markersize=8)
    
    colors = ['blue', 'orange', 'pink']
    linestyles = ['-.', ':', '-']
    for i, (model_name, values) in enumerate(baseline_values.items()):
        valid_positions = [pos for pos, val in zip(positions, values) if val is not None]
        valid_values = [val for val in values if val is not None]
        if valid_values:
            plt.plot(valid_positions, valid_values, color=colors[i], linestyle=linestyles[i], 
                    marker='o', label=model_name.replace('_', ' ').title(), linewidth=2, markersize=6)
    
    if has_custom_heuristic:
        plt.plot(positions, custom_heuristic_values, 'r-s', 
                label='Custom Heuristic', linewidth=2, markersize=8)
    
    plt.xticks(positions, testcase_labels, rotation=45, ha='right')
    plt.xlabel('Test Case')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Comparison: All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_filename = f'{metric}_comparison_plot.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.close()

def create_model_performance_summary(comparison_results, congestion_df, output_dir, has_traditional=True):
    print(f"\nGenerating comprehensive model performance summary...")
    
    model_types = ['finetune_model', 'combined_model', 'random_forest', 'knn', 'decision_tree']
    if has_traditional:
        model_types.append('traditional_best')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    similarity_metrics = ['throughput_similarity_to_gt', 'distribution_similarity_to_gt', 'overlap_with_gt_range_pct']
    
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
        
        # Calculate performance metrics
        model_metrics = {}
        for metric in metrics:
            values = []
            for result in comparison_results:
                if model_type == 'traditional_best':
                    if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                        if result.get('traditional_best', {}).get('exists', False):
                            trad_result = result['traditional_best']['metrics']
                            if metric in trad_result:
                                values.append(float(trad_result[metric]))
                elif model_type in result and 'metrics' in result[model_type]:
                    values.append(float(result[model_type]['metrics'][metric]))
                elif model_type == 'combined_model' and not result.get('combined_model', {}).get('exists', True):
                    continue
            
            if values:
                model_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        # Calculate similarity metrics (only for ML models)
        similarity_stats = {}
        if model_type != 'traditional_best' and congestion_df is not None:
            for sim_metric in similarity_metrics:
                col_name = f'{model_type}_{sim_metric}'
                if col_name in congestion_df.columns:
                    values = congestion_df[col_name].fillna(0).values
                    valid_values = values[values > 0]  # Exclude cases with no predictions
                    if len(valid_values) > 0:
                        similarity_stats[sim_metric] = {
                            'mean': np.mean(valid_values),
                            'std': np.std(valid_values),
                            'count': len(valid_values)
                        }
        
        # Create summary entry
        if any(model_metrics.get(m, {}).get('count', 0) > 0 for m in model_metrics):
            summary_entry = {
                'model': model_label,
                'model_type': model_type,
                'test_cases': max(model_metrics.get(m, {}).get('count', 0) for m in model_metrics)
            }
            
            # Add performance metrics
            for metric in metrics:
                if metric in model_metrics and model_metrics[metric]['count'] > 0:
                    summary_entry[f'{metric}_mean'] = model_metrics[metric]['mean']
                    summary_entry[f'{metric}_std'] = model_metrics[metric]['std']
                else:
                    summary_entry[f'{metric}_mean'] = 0
                    summary_entry[f'{metric}_std'] = 0
            
            # Add similarity metrics
            for sim_metric in similarity_metrics:
                if sim_metric in similarity_stats:
                    summary_entry[f'{sim_metric}_mean'] = similarity_stats[sim_metric]['mean']
                    summary_entry[f'{sim_metric}_std'] = similarity_stats[sim_metric]['std']
                else:
                    summary_entry[f'{sim_metric}_mean'] = 0
                    summary_entry[f'{sim_metric}_std'] = 0
            
            summary_data.append(summary_entry)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'comprehensive_model_performance_summary.csv'), index=False)
    
    # Create enhanced heatmap including similarity metrics
    if len(summary_data) > 0:
        # Performance heatmap
        perf_data = []
        model_labels = []
        
        for row in summary_data:
            if row['test_cases'] > 0:
                if row['model_type'] == 'traditional_best':
                    metric_values = [
                        row['accuracy_mean'], row['precision_mean'], 
                        row['recall_mean'], row['f1_score_mean'], 0.0
                    ]
                else:
                    metric_values = [row[f'{metric}_mean'] for metric in metrics]
                
                perf_data.append(metric_values)
                model_labels.append(row['model'])
        
        if perf_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(model_labels) * 0.5)))
            
            # Performance heatmap
            perf_array = np.array(perf_data)
            im1 = ax1.imshow(perf_array, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
            
            ax1.set_xticks(np.arange(len(metrics)))
            ax1.set_yticks(np.arange(len(model_labels)))
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.set_yticklabels(model_labels)
            
            for i in range(len(model_labels)):
                for j in range(len(metrics)):
                    value = perf_array[i, j]
                    if j == 4 and model_labels[i] == 'Traditional Heuristic' and value == 0:
                        text_val = "N/A"
                        color = "gray"
                    else:
                        text_val = f'{value:.3f}'
                        color = "black"
                    
                    ax1.text(j, i, text_val, ha="center", va="center", 
                            color=color, fontweight='bold')
            
            ax1.set_title("Model Performance Metrics")
            
            # Similarity heatmap (exclude traditional model)
            ml_models = [row for row in summary_data if row['model_type'] != 'traditional_best' and row['test_cases'] > 0]
            if ml_models:
                sim_data = []
                sim_labels = []
                
                for row in ml_models:
                    sim_values = [row[f'{metric}_mean'] for metric in similarity_metrics]
                    sim_data.append(sim_values)
                    sim_labels.append(row['model'])
                
                sim_array = np.array(sim_data)
                im2 = ax2.imshow(sim_array, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
                
                ax2.set_xticks(np.arange(len(similarity_metrics)))
                ax2.set_yticks(np.arange(len(sim_labels)))
                ax2.set_xticklabels(['Throughput\nSimilarity', 'Distribution\nSimilarity', 'Overlap %'])
                ax2.set_yticklabels(sim_labels)
                
                for i in range(len(sim_labels)):
                    for j in range(len(similarity_metrics)):
                        value = sim_array[i, j]
                        if j == 2:  # Overlap percentage
                            text_val = f'{value:.1f}%'
                        else:
                            text_val = f'{value:.3f}'
                        ax2.text(j, i, text_val, ha="center", va="center", 
                                color="black", fontweight='bold')
                
                ax2.set_title("Congestion Similarity Metrics")
                
                plt.colorbar(im2, ax=ax2)
            else:
                ax2.set_visible(False)
            
            plt.colorbar(im1, ax=ax1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'comprehensive_performance_heatmap.png'), dpi=300, bbox_inches='tight')
            print(f"Comprehensive performance heatmap saved")
            plt.close()
    
    return summary_df

def main():
    results_dir = 'results'
    models_dir = os.path.join(results_dir, 'models')
    output_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    custom_heuristic_params = {
        'short_window': 200,       
        'long_window': 28500,      
        'peak_tolerance': 0.3,  
        'count_threshold': 3    
    }
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION WITH ENHANCED CONGESTION ANALYSIS")
    print("="*80)
    
    print("\nLoading dataset...")
    data = pd.read_csv('combined_features.csv')
    print(f"Loaded {len(data)} records")
    
    param_tests_file = os.path.abspath('combined_param_tests.csv')
    if not os.path.exists(param_tests_file):
        print(f"Warning: Traditional parameter test file not found")
        has_traditional = False
        param_df = None
    else:
        print(f"Loading traditional parameter test results")
        param_df = pd.read_csv(param_tests_file)
        has_traditional = True
    
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
    print(f"Found {len(unique_testcases)} unique test cases")
    
    # Load models
    finetune_model_path = os.path.join(models_dir, 'finetune_final_model.pt')
    if not os.path.exists(finetune_model_path):
        finetune_files = [f for f in os.listdir(models_dir) if f.startswith('finetune_step') and f.endswith('.pt')]
        if finetune_files:
            step_nums = [int(re.search(r'step(\d+)', f).group(1)) for f in finetune_files]
            highest_step = max(step_nums)
            finetune_model_path = os.path.join(models_dir, [f for f in finetune_files if f'step{highest_step}' in f][0])

    print(f"Loading finetune model...")
    finetune_state_dict, finetune_time_steps = load_model(finetune_model_path)
    
    combined_model_path = os.path.join(models_dir, 'combined_model_downstream_only.pt')
    if not os.path.exists(combined_model_path):
        print("Warning: Combined model not found")
        has_combined = False
    else:
        print(f"Loading combined model...")
        combined_state_dict, combined_time_steps = load_model(combined_model_path)
        has_combined = True
    
    baseline_models = {}
    baseline_model_names = ['random_forest', 'knn', 'decision_tree']
    
    for model_name in baseline_model_names:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            model, feature_names = load_baseline_model(model_path)
            if model is not None:
                baseline_models[model_name] = {'model': model, 'feature_names': feature_names}
                print(f"Loaded baseline model: {model_name}")
    
    print(f"Loaded {len(baseline_models)} baseline models")
    
    # Run model evaluations
    comparison_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nRunning model evaluations...")
    for testcase in unique_testcases:
        filename, network_limit = testcase.rsplit('_', 1)
        network_limit = int(network_limit)
        
        if 'fhd' in filename.lower():
            continue
            
        scenario = extract_scenario_name(filename)
        
        test_data = downstream_data[
            (downstream_data['file_name'] == filename) & 
            (downstream_data['network_limit'] == network_limit)
        ]
        
        if len(test_data) <= 15:
            continue
        
        results_entry = {
            'testcase': testcase,
            'scenario': scenario,
            'network_limit': network_limit,
            'sample_count': int(len(test_data))
        }
        
        # Evaluate all models for this testcase
        X_test_seq, y_test_seq, feature_cols, _ = preprocess_data(
            test_data, include_network_limit=True, time_steps=finetune_time_steps
        )
        finetune_model = LSTMModel(input_size=X_test_seq.shape[2])
        finetune_model.load_state_dict(finetune_state_dict)
        finetune_model.to(device)
        finetune_metrics = evaluate_model(finetune_model, X_test_seq, y_test_seq)
        results_entry['finetune_model'] = {'metrics': finetune_metrics}
        
        if has_combined:
            X_test_seq, y_test_seq, feature_cols, _ = preprocess_data(
                test_data, include_network_limit=True, time_steps=combined_time_steps
            )
            combined_model = LSTMModel(input_size=X_test_seq.shape[2])
            combined_model.load_state_dict(combined_state_dict)
            combined_model.to(device)
            combined_metrics = evaluate_model(combined_model, X_test_seq, y_test_seq)
            results_entry['combined_model'] = {'metrics': combined_metrics}
        else:
            results_entry['combined_model'] = {'exists': False}
        
        if baseline_models:
            X_test_baseline, y_test_baseline, _, _ = preprocess_data_baseline(
                test_data, include_network_limit=True
            )
            
            for model_name, model_info in baseline_models.items():
                try:
                    model = model_info['model']
                    baseline_metrics = evaluate_baseline_model(model, X_test_baseline, y_test_baseline)
                    results_entry[model_name] = {'metrics': baseline_metrics}
                except Exception as e:
                    results_entry[model_name] = {'error': str(e)}
        
        if has_traditional:
            traditional_best = get_traditional_best_accuracy(param_tests_file, testcase)
            if traditional_best:
                results_entry['traditional_best'] = {'exists': True, 'metrics': traditional_best}
            else:
                results_entry['traditional_best'] = {'exists': False}
        
        comparison_results.append(results_entry)
    
    # ENHANCED CONGESTION ANALYSIS
    print("\n" + "="*80)
    print("ENHANCED CONGESTION THROUGHPUT ANALYSIS")
    print("="*80)
    congestion_df = analyze_congestion_throughput_improved(comparison_results, data, output_dir)
    
    # Generate all visualizations
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    if congestion_df is not None:
        create_congestion_throughput_plots(congestion_df, output_dir)
    
    available_models = ['finetune']
    if has_combined:
        available_models.append('combined')
    for model_name in baseline_model_names:
        if any(model_name in result for result in comparison_results):
            available_models.append(model_name)
    
    create_roc_curve_plot(comparison_results, output_dir, available_models)
    
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='accuracy', has_combined=has_combined, has_traditional=has_traditional,
                          custom_heuristic_params=custom_heuristic_params)
    
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='f1_score', has_combined=has_combined, has_traditional=has_traditional,
                          custom_heuristic_params=custom_heuristic_params)
    
    create_comparison_plot(comparison_results, param_df if has_traditional else None, output_dir, 
                          metric='roc_auc', has_combined=has_combined, has_traditional=False)
    
    # Save comprehensive results
    summary_data = []
    for result in comparison_results:
        row = {
            'testcase': result['testcase'],
            'scenario': result['scenario'],
            'network_limit': result['network_limit'],
            'sample_count': result['sample_count'],
        }
        
        if 'finetune_model' in result:
            finetune_metrics = result['finetune_model']['metrics']
            row.update({
                'finetune_accuracy': float(finetune_metrics['accuracy']),
                'finetune_f1': float(finetune_metrics['f1_score']),
                'finetune_auc': float(finetune_metrics['roc_auc']),
            })
        
        if has_combined and result.get('combined_model', {}).get('exists', True):
            comb_metrics = result['combined_model']['metrics']
            row.update({
                'combined_accuracy': float(comb_metrics['accuracy']),
                'combined_f1': float(comb_metrics['f1_score']),
                'combined_auc': float(comb_metrics['roc_auc']),
            })
        
        for model_name in baseline_model_names:
            if model_name in result and 'metrics' in result[model_name]:
                metrics = result[model_name]['metrics']
                row.update({
                    f'{model_name}_accuracy': float(metrics['accuracy']),
                    f'{model_name}_f1': float(metrics['f1_score']),
                    f'{model_name}_auc': float(metrics['roc_auc']),
                })
        
        if has_traditional and result.get('traditional_best', {}).get('exists', False):
            trad_metrics = result['traditional_best']['metrics']
            row.update({
                'traditional_accuracy': float(trad_metrics['accuracy']),
                'traditional_f1': float(trad_metrics['f1_score']),
            })
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'comprehensive_model_comparison.csv'), index=False)
    
    # Save detailed JSON results
    with open(os.path.join(output_dir, 'full_comparison_results_with_baselines.json'), 'w') as f:
        json_ready_results = []
        for result in comparison_results:
            result_copy = result.copy()
            result_copy['network_limit'] = int(result_copy['network_limit'])
            result_copy['sample_count'] = int(result_copy['sample_count'])
            
            for model_key in result_copy.keys():
                if isinstance(result_copy[model_key], dict) and 'metrics' in result_copy[model_key]:
                    metrics = result_copy[model_key]['metrics']
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            metrics[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            metrics[key] = float(value)
            
            json_ready_results.append(result_copy)
        
        json.dump(json_ready_results, f, indent=2)
    
    # Generate final comprehensive summary
    summary_df = create_model_performance_summary(comparison_results, congestion_df, output_dir, has_traditional)
    
    # FINAL SUMMARY OUTPUT
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - ALL FILES GENERATED")
    print("="*80)
    
    print(f"\n📊 CSV REPORTS GENERATED:")
    print(f"  • comprehensive_model_comparison.csv - Main results by testcase")
    print(f"  • congestion_throughput_analysis.csv - Enhanced congestion analysis")
    print(f"  • comprehensive_model_performance_summary.csv - Performance stats + similarity")
    print(f"  • full_comparison_results_with_baselines.json - Complete detailed results")
    
    print(f"\n📈 FIGURES GENERATED:")
    print(f"  • ROC Analysis: roc_curves_individual.png, roc_curves_aggregate.png")
    print(f"  • Performance Comparisons: accuracy_comparison_plot.png, f1_score_comparison_plot.png, roc_auc_comparison_plot.png")
    print(f"  • Original Congestion: congestion_throughput_comparison.png, congestion_capacity_utilization.png")
    print(f"  • NEW Similarity Analysis: throughput_similarity_comparison.png, throughput_overlap_comparison.png, distribution_similarity_comparison.png")
    print(f"  • Performance Summary: comprehensive_performance_heatmap.png")
    
    if congestion_df is not None and not congestion_df.empty:
        print(f"\n🔍 ENHANCED ANALYSIS INSIGHTS:")
        print(f"The improved analysis answers: 'Do models detect congestion when throughput matches ground truth?'")
        
        models_analyzed = ['finetune_model', 'combined_model', 'random_forest', 'knn', 'decision_tree']
        
        print(f"\nModel Throughput Similarity Summary:")
        for model in models_analyzed:
            sim_col = f'{model}_throughput_similarity_to_gt'
            overlap_col = f'{model}_overlap_with_gt_range_pct'
            
            if sim_col in congestion_df.columns and overlap_col in congestion_df.columns:
                valid_sim = congestion_df[congestion_df[f'{model}_congested_samples'] > 0][sim_col]
                valid_overlap = congestion_df[congestion_df[f'{model}_congested_samples'] > 0][overlap_col]
                
                if len(valid_sim) > 0:
                    model_name = model.replace('_model', '').replace('_', ' ').title()
                    if model == 'finetune_model':
                        model_name = 'Finetune Model'
                    elif model == 'combined_model':
                        model_name = 'Combined Model'
                    
                    print(f"  {model_name}: Similarity={valid_sim.mean():.3f}, Overlap={valid_overlap.mean():.1f}%")
    
    print(f"\n✅ All analysis complete! Check the '{output_dir}' folder for all results.")

if __name__ == "__main__":
    main()