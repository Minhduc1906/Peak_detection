import pandas as pd
import numpy as np
import os
import json
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Create directory structure - same as LSTM file
results_dir = 'results'
for subdir in ['models', 'plots', 'metrics', 'time']:
    os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

# Global list to store timing data
training_times = []

# Helper functions (same as LSTM file)
def extract_scenario_name(filename):
    base_name = filename.replace('.csv', '').replace('test_', '')
    parts = base_name.split('_')
    scenario_parts = []
    for part in parts:
        if any(keyword in part for keyword in ['limited', 'unlimited', 'mbps', 'load', 'randomized']):
            break
        scenario_parts.append(part)
    return '_'.join(scenario_parts)

def save_timing_data():
    """Save timing data to CSV file"""
    if training_times:
        timing_df = pd.DataFrame(training_times)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, 'time', f'baseline_times_{timestamp}.csv')
        timing_df.to_csv(csv_path, index=False)
        print(f"\nBaseline timing data saved to: {csv_path}")
        
        # Also save a summary CSV with just the latest run
        summary_path = os.path.join(results_dir, 'time', 'latest_baseline_times.csv')
        timing_df.to_csv(summary_path, index=False)
        print(f"Latest baseline timing summary saved to: {summary_path}")

def preprocess_data_baseline(df, include_filename_features=False, include_direction=False, 
                           include_network_limit=True, scaler=None):
    """
    Preprocess data for baseline models (no time sequences needed)
    """
    data_copy = df.copy()
    data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_copy.fillna(data_copy.max(), inplace=True)
    
    drop_cols = ['queue_size']
    if not include_filename_features:
        drop_cols.extend(['file_name', 'file_name_encoded'])
    else:
        drop_cols.append('file_name')
    
    if not include_direction and 'direction' in data_copy.columns:
        drop_cols.append('direction')
    
    if not include_network_limit and 'network_limit' in data_copy.columns:
        drop_cols.append('network_limit')
        
    for col in drop_cols:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
    
    # Handle network_limit normalization
    if include_network_limit and 'network_limit' in data_copy.columns:
        limits = data_copy['network_limit'].unique()
        if len(limits) > 1:
            max_limit = data_copy['network_limit'].max()
            min_limit = data_copy['network_limit'].min()
            if max_limit > min_limit:
                data_copy['network_limit'] = (data_copy['network_limit'] - min_limit) / (max_limit - min_limit)
    
    # Drop additional columns
    for col in ['subfolder', 'testcase_with_limit']:
        if col in data_copy.columns:
            data_copy = data_copy.drop(columns=[col])
    
    feature_cols = [c for c in data_copy.columns if c not in ['queue_exists', 'detection_accuracy', 'time', 'relative_time', 'network_limit']]
    
    X = data_copy[feature_cols].values
    y = data_copy['queue_exists'].values
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return comprehensive metrics"""
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities (handle models that may not have predict_proba)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
            # Normalize decision function to [0,1] range
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            y_prob = y_pred.astype(float)  # Fallback to predictions
    except:
        y_prob = y_pred.astype(float)  # Fallback to predictions
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        # Handle case where all labels are the same class
        roc_auc = 0.5
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred.tolist(),
        'targets': y_test.tolist(),
        'probabilities': y_prob.tolist()
    }

def train_and_evaluate_baseline(X_train, X_test, y_train, y_test, model, model_name, 
                              feature_names=None):
    """
    Train and evaluate a baseline model
    """
    # Track training start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\nStarting training for {model_name} at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Track training end time
    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0
    
    print(f"Completed training for {model_name} at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")
    
    # Evaluate the model
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)
    
    print(f'{model_name} - Train Acc: {train_metrics["accuracy"]:.4f}, Test Acc: {test_metrics["accuracy"]:.4f}')
    print(f'{model_name} - Train AUC: {train_metrics["roc_auc"]:.4f}, Test AUC: {test_metrics["roc_auc"]:.4f}')
    
    # Store timing data
    timing_data = {
        'model_name': model_name,
        'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(duration_seconds, 2),
        'duration_minutes': round(duration_minutes, 2),
        'training_samples': len(X_train),
        'validation_samples': len(X_test),
        'final_train_accuracy': round(train_metrics['accuracy'], 4),
        'final_test_accuracy': round(test_metrics['accuracy'], 4),
        'final_train_auc': round(train_metrics['roc_auc'], 4),
        'final_test_auc': round(test_metrics['roc_auc'], 4),
        'model_type': 'baseline'
    }
    training_times.append(timing_data)
    
    # Prepare metrics dictionary
    display_name = model_name.replace("_", " ").title()
    metrics = {
        'model_name': model_name,
        'display_name': display_name,
        'train_accuracy': train_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'train_auc': train_metrics['roc_auc'],
        'test_auc': test_metrics['roc_auc'],
        'train_precision': train_metrics['precision'],
        'test_precision': test_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'test_recall': test_metrics['recall'],
        'train_f1': train_metrics['f1_score'],
        'test_f1': test_metrics['f1_score'],
        'feature_names': feature_names,
        'training_duration_seconds': duration_seconds,
        'training_duration_minutes': duration_minutes,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    # Save model to models folder
    model_path = os.path.join(results_dir, 'models', f"{model_name}.joblib")
    joblib.dump({
        'model': model,
        'metrics': metrics,
        'feature_names': feature_names,
        'training_time': timing_data
    }, model_path)
    
    # Save metrics JSON to metrics folder
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = metrics.copy()
    metrics_json['train_metrics'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in train_metrics.items()}
    metrics_json['test_metrics'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in test_metrics.items()}
    
    with open(os.path.join(results_dir, 'metrics', f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Create performance plot
    create_performance_plot(model_name, train_metrics, test_metrics)
    
    # Feature importance plot (if model supports it)
    if hasattr(model, 'feature_importances_') and feature_names:
        create_feature_importance_plot(model, feature_names, model_name)
    
    return model, metrics

def create_performance_plot(model_name, train_metrics, test_metrics):
    """Create a performance comparison plot for train vs test"""
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    train_values = [train_metrics[m] for m in metrics_names]
    test_values = [test_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name.replace("_", " ").title()} - Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', f"{model_name}_performance.png"))
    plt.close()

def create_feature_importance_plot(model, feature_names, model_name):
    """Create feature importance plot for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Get top 20 features
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'{model_name.replace("_", " ").title()} - Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[top_indices])
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', f"{model_name}_feature_importance.png"))
    plt.close()

def create_comparison_plot(all_metrics):
    """Create a comparison plot across all baseline models"""
    model_names = [m['display_name'] for m in all_metrics]
    test_accuracies = [m['test_accuracy'] for m in all_metrics]
    test_aucs = [m['test_auc'] for m in all_metrics]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    bars1 = ax1.bar(x, test_accuracies, width, alpha=0.8, color='skyblue')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison - Baseline Models')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # AUC plot
    bars2 = ax2.bar(x, test_aucs, width, alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Test AUC')
    ax2.set_title('Test AUC Comparison - Baseline Models')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'baseline_models_comparison.png'))
    plt.close()

def main():
    # Track overall execution start time
    main_start_time = time.time()
    main_start_datetime = datetime.now()
    print(f"Starting baseline models training at {main_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("Loading data...")
    data = pd.read_csv('combined_features.csv')
    print(f"Loaded {len(data)} records")
    
    # Add network_limit column if missing
    if 'network_limit' not in data.columns and 'subfolder' in data.columns:
        data['network_limit'] = data['subfolder'].apply(
            lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1
        )
    elif 'network_limit' not in data.columns:
        data['network_limit'] = -1
    
    # Focus on downstream data like the LSTM model
    downstream_data = data[data['direction'] == 'downstream'].copy()
    print(f"Using {len(downstream_data)} downstream records for training")
    
    # Encode filename
    filename_encoder = LabelEncoder()
    data['file_name_encoded'] = filename_encoder.fit_transform(data['file_name'])
    downstream_data['file_name_encoded'] = filename_encoder.transform(downstream_data['file_name'])
    
    # Preprocess data for baseline models
    print("\nPreprocessing data for baseline models...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data_baseline(
        downstream_data, 
        include_filename_features=False, 
        include_direction=False,
        include_network_limit=True
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
    
    # Define baseline models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5, 
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=6, 
            random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=10, 
            random_state=42
        )
    }
    
    all_metrics = []
    
    # Train each baseline model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        try:
            trained_model, metrics = train_and_evaluate_baseline(
                X_train, X_test, y_train, y_test, 
                model, model_name, feature_names
            )
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Create comparison plots
    if all_metrics:
        print(f"\nCreating comparison plots...")
        create_comparison_plot(all_metrics)
        
        # Save summary metrics
        summary_data = []
        for metrics in all_metrics:
            summary_data.append({
                'model_name': metrics['model_name'],
                'display_name': metrics['display_name'],
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'train_auc': metrics['train_auc'],
                'test_auc': metrics['test_auc'],
                'train_f1': metrics['train_f1'],
                'test_f1': metrics['test_f1'],
                'training_duration_minutes': metrics['training_duration_minutes']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, 'metrics', 'baseline_models_summary.csv'), index=False)
        
        # Save detailed metrics
        with open(os.path.join(results_dir, 'metrics', 'baseline_models_detailed.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_ready_metrics = []
            for metrics in all_metrics:
                metrics_copy = metrics.copy()
                # Handle nested metrics dictionaries
                for key in ['train_metrics', 'test_metrics']:
                    if key in metrics_copy:
                        metrics_copy[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in metrics_copy[key].items()}
                json_ready_metrics.append(metrics_copy)
            
            json.dump(json_ready_metrics, f, indent=4)
    
    # Track overall execution end time
    main_end_time = time.time()
    main_end_datetime = datetime.now()
    main_duration_seconds = main_end_time - main_start_time
    main_duration_minutes = main_duration_seconds / 60.0
    
    print(f"\nCompleted baseline models training at {main_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution duration: {main_duration_minutes:.2f} minutes ({main_duration_seconds:.2f} seconds)")
    
    # Add overall execution timing
    overall_execution_timing = {
        'model_name': 'baseline_overall_execution',
        'start_time': main_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': main_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(main_duration_seconds, 2),
        'duration_minutes': round(main_duration_minutes, 2),
        'total_models_trained': len(all_metrics),
        'model_type': 'baseline_execution'
    }
    training_times.append(overall_execution_timing)
    
    # Save timing data
    save_timing_data()
    
    print(f"\nBaseline models training complete!")
    print(f"Models saved to: {os.path.join(results_dir, 'models')}")
    print(f"Metrics saved to: {os.path.join(results_dir, 'metrics')}")
    print(f"Plots saved to: {os.path.join(results_dir, 'plots')}")
    print(f"Timing data saved to: {os.path.join(results_dir, 'time')}")
    
    # Print summary
    if all_metrics:
        print(f"\n{'='*60}")
        print("BASELINE MODELS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Test Acc':<10} {'Test AUC':<10} {'Time (min)':<12}")
        print("-" * 60)
        for metrics in all_metrics:
            print(f"{metrics['display_name']:<20} {metrics['test_accuracy']:<10.3f} {metrics['test_auc']:<10.3f} {metrics['training_duration_minutes']:<12.2f}")

if __name__ == "__main__":
    main()